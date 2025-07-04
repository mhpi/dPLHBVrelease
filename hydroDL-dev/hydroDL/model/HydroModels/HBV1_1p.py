import torch
from hydroDL.model.rnn import UH_gamma, UH_conv


"""HBV1.1p Model used in Song et al. 2025: https://doi.org/10.22541/essoar.172304428.82707157/v2"""

class HBV(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBV, self).__init__()

    def forward(self, x, parameters, staind, tdlst, mu, muwts, rtwts, bufftime=0, outstate=False, routOpt=False,
                comprout=False, dydrop=False):
        # Modified from the original numpy version from Beck et al., 2020. (http://www.gloh2o.org/hbv/) which
        # runs the HBV-light hydrological model (Seibert, 2005).
        # NaN values have to be removed from the inputs.
        #
        # Input:
        #     X: dim=[time, basin, var] forcing array with var P(mm/d), T(deg C), PET(mm/d)
        #     parameters: array with parameter values having the following structure and scales:
        #         BETA[1,6]; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         PERC[0,10]; UZL[0,100]; TT[-2.5,2.5]; CFMAX[0.5,10]; CFR[0,0.1]; CWH[0,0.2];BETAT[0.3,5]; parC[0,1];
        #     staind:use which time step from the learned para time series for static parameters
        #     tdlst: the index list of hbv parameters set as dynamic
        #     mu:number of components; muwts: weights of components if True; rtwts: routing parameters;
        #     bufftime:warm up period; outstate: output state var; routOpt:routing option; comprout:component routing opt
        #     dydrop: the possibility to drop a dynamic para to static to reduce potential overfitting
        #
        #
        # Output, all in mm:
        #     outstate True: output most state variables for warm-up
        #      Qs:simulated streamflow; SNOWPACK:snow depth; MELTWATER:snow water holding depth;
        #      SM:soil storage; SUZ:upper zone storage; SLZ:lower zone storage
        #     outstate False: output the simulated flux array Qall contains
        #      Qs:simulated streamflow=Q0+Q1+Q2; Qsimave0:Q0 component; Qsimave1:Q1 component; Qsimave2:Q2 baseflow componnet
        #      ETave: actual ET

        PRECS = 1e-5  # keep the numerical calculation stable

        # Initialization to warm-up states
        if bufftime > 0:
            with torch.no_grad():
                xinit = x[0:bufftime, :, :]
                initmodel = HBVMulET_1_1p()
                buffpara = parameters[bufftime-1, :, :, :]
                Qsinit, SNOWPACK, MELTWATER, SM, SUZ, SLZ = initmodel(xinit, buffpara, mu, muwts, rtwts,
                                                                      bufftime=0, outstate=True, routOpt=False, comprout=False)
        else:

            # Without warm-up bufftime=0, initialize state variables with zeros
            Ngrid = x.shape[1]
            SNOWPACK = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            MELTWATER = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            SM = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            SUZ = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            SLZ = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()

        P = x[bufftime:, :, 0]
        Pm= P.unsqueeze(2).repeat(1,1,mu) # precip
        T = x[bufftime:, :, 1]
        Tm = T.unsqueeze(2).repeat(1,1,mu) # temperature
        ETpot = x[bufftime:, :, 2]
        ETpm = ETpot.unsqueeze(2).repeat(1,1,mu) # potential ET
        parAll = parameters[bufftime:, :, :, :]
        parAllTrans = torch.zeros_like(parAll)

        ## scale the parameters to real values from [0,1]
        hbvscaLst = [[1,6], [50,1000], [0.05,0.9], [0.01,0.5], [0.001,0.2], [0.2,1],
                        [0,10], [0,100], [-2.5,2.5], [0.5,10], [0,0.1], [0,0.2], [0.3,5],[0,1]]  # HBV para
        routscaLst = [[0,2.9], [0,6.5]]  # routing para

        for ip in range(len(hbvscaLst)): # not include routing. Scaling the parameters
            parAllTrans[:,:,ip,:] = hbvscaLst[ip][0] + parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])

        Nstep, Ngrid = P.size()

        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        parstaFull = parAllTrans[staind, :, :, :].unsqueeze(0).repeat([Nstep, 1, 1, 1])  # static para matrix
        parhbvFull = torch.clone(parstaFull)
        # create probability mask for each parameter on the basin dimension
        pmat = torch.ones([1, Ngrid, 1])*dydrop
        for ix in tdlst:
            staPar = parstaFull[:, :, ix-1, :]
            dynPar = parAllTrans[:, :, ix-1, :]
            drmask = torch.bernoulli(pmat).detach_().cuda()  # to drop dynamic parameters as static in some basins
            comPar = dynPar*(1-drmask) + staPar*drmask
            parhbvFull[:, :, ix-1, :] = comPar


        # Initialize time series of model variables to save results
        Qsimmu = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()
        ETmu = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()

        # Output the box components of Q
        Qsimmu0 = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()
        Qsimmu1 = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()
        Qsimmu2 = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()

        # # Not used. Logging the state variables for debug.
        # # SMlog = np.zeros(P.size())
        # logSM = np.zeros(P.size())
        # logPS = np.zeros(P.size())
        # logswet = np.zeros(P.size())
        # logRE = np.zeros(P.size())

        for t in range(Nstep):
            paraLst = []
            for ip in range(len(hbvscaLst)):  # unpack HBV parameters
                paraLst.append(parhbvFull[t, :, ip, :])

            parBETA, parFC, parK0, parK1, parK2, parLP, parPERC, parUZL, parTT, parCFMAX, parCFR, parCWH, parBETAET,parC = paraLst
            # Separate precipitation into liquid and solid components
            PRECIP = Pm[t, :, :]
            RAIN = torch.mul(PRECIP, (Tm[t, :, :] >= parTT).type(torch.float32))
            SNOW = torch.mul(PRECIP, (Tm[t, :, :] < parTT).type(torch.float32))

            # Snow process
            SNOWPACK = SNOWPACK + SNOW
            melt = parCFMAX * (Tm[t, :, :] - parTT)
            melt = torch.clamp(melt, min=0.0)
            melt = torch.min(melt, SNOWPACK)
            MELTWATER = MELTWATER + melt
            SNOWPACK = SNOWPACK - melt
            refreezing = parCFR * parCFMAX * (parTT - Tm[t, :, :])
            refreezing = torch.clamp(refreezing, min=0.0)
            refreezing = torch.min(refreezing, MELTWATER)
            SNOWPACK = SNOWPACK + refreezing
            MELTWATER = MELTWATER - refreezing
            tosoil = MELTWATER - (parCWH * SNOWPACK)
            tosoil = torch.clamp(tosoil, min=0.0)
            MELTWATER = MELTWATER - tosoil

            # Soil and evaporation
            soil_wetness = (SM / parFC) ** parBETA
            soil_wetness = torch.clamp(soil_wetness, min=0.0, max=1.0)
            recharge = (RAIN + tosoil) * soil_wetness

            # Not used, logging states for checking
            # logSM[t,:] = SM.detach().cpu().numpy()
            # logPS[t,:] = (RAIN + tosoil).detach().cpu().numpy()
            # logswet[t,:] = (SM / parFC).detach().cpu().numpy()
            # logRE[t, :] = recharge.detach().cpu().numpy()

            SM = SM + RAIN + tosoil - recharge
            excess = SM - parFC
            excess = torch.clamp(excess, min=0.0)
            SM = SM - excess
            # MODIFY here. Different as class HBVMulT(). Add a ET shape parameter parBETAET
            evapfactor = (SM / (parLP * parFC)) ** parBETAET
            evapfactor  = torch.clamp(evapfactor, min=0.0, max=1.0)
            ETact = ETpm[t, :, :] * evapfactor
            ETact = torch.min(SM, ETact)
            SM = torch.clamp(SM - ETact, min=PRECS) # SM can not be zero for gradient tracking

            capillary = torch.min(SLZ, parC * SLZ * (1.0 - torch.clamp(SM / parFC, max=1.0)))

            SM = torch.clamp(SM + capillary, min=PRECS)
            SLZ = torch.clamp(SLZ - capillary, min=PRECS)


            # Groundwater boxes
            SUZ = SUZ + recharge + excess
            PERC = torch.min(SUZ, parPERC)
            SUZ = SUZ - PERC
            Q0 = parK0 * torch.clamp(SUZ - parUZL, min=0.0)
            SUZ = SUZ - Q0
            Q1 = parK1 * SUZ
            SUZ = SUZ - Q1
            SLZ = SLZ + PERC
            Q2 = parK2 * SLZ
            SLZ = SLZ - Q2
            Qsimmu[t, :, :] = Q0 + Q1 + Q2

            # save components for Q
            Qsimmu0[t, :, :] = Q0
            Qsimmu1[t, :, :] = Q1
            Qsimmu2[t, :, :] = Q2
            ETmu[t, :, :] = ETact

            # Not used, for debug state variables
            # SMlog[t,:, :] = SM.detach().cpu().numpy()
            # SUZlog[t,:,:] = SUZ.detach().cpu().numpy()
            # SLZlog[t,:,:] = SLZ.detach().cpu().numpy()

        Qsimave0 = Qsimmu0.mean(-1, keepdim=True)
        Qsimave1 = Qsimmu1.mean(-1, keepdim=True)
        Qsimave2 = Qsimmu2.mean(-1, keepdim=True)
        ETave = ETmu.mean(-1, keepdim=True)

        # get the initial average
        if muwts is None: # simple average
            Qsimave = Qsimmu.mean(-1)
        else: # weighted average using learned weights
            Qsimave = (Qsimmu * muwts).sum(-1)

        if routOpt is True: # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim = Qsimmu.view(Nstep, Ngrid * mu)
            else:
                # average the components, then do routing
                Qsim = Qsimave

            # scale two routing parameters
            tempa = routscaLst[0][0] + rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb = routscaLst[1][0] + rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])
            routa = tempa.repeat(Nstep, 1).unsqueeze(-1)
            routb = tempb.repeat(Nstep, 1).unsqueeze(-1)
            UH = UH_gamma(routa, routb, lenF=15)  # lenF: folter
            rf = torch.unsqueeze(Qsim, -1).permute([1, 2, 0])   # dim:gage*var*time
            UH = UH.permute([1, 2, 0])  # dim: gage*var*time
            Qsrout = UH_conv(rf, UH).permute([2, 0, 1])

            if comprout is True: # Qs is [time, [gage*mult], var] now
                Qstemp = Qsrout.view(Nstep, Ngrid, mu)
                if muwts is None:
                    Qs = Qstemp.mean(-1, keepdim=True)
                else:
                    Qs = (Qstemp * muwts).sum(-1, keepdim=True)
            else:
                Qs = Qsrout

        else: # no routing, output the initial average simulations
            Qs = torch.unsqueeze(Qsimave, -1) # add a dimension

        if outstate is True: # output states
            return Qs, SNOWPACK, MELTWATER, SM, SUZ, SLZ
        else:
            # return Qs
            Qall = torch.cat((Qs, Qsimave0, Qsimave1, Qsimave2, ETave), dim=-1)
            return Qall




class HBVMulET_1_1p(torch.nn.Module):
    """Multi-component HBV Model PyTorch version"""
    # Add an ET shape parameter; others are the same as class HBVMul()
    # refer HBVMul() for detailed comments

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMulET_1_1p, self).__init__()

    def forward(self, x, parameters, mu, muwts, rtwts, bufftime=0, outstate=False, routOpt=False, comprout=False):

        PRECS = 1e-5

        # Initialization
        if bufftime > 0:
            with torch.no_grad():
                xinit = x[0:bufftime, :, :]
                initmodel = HBVMulET_1_1p()
                Qsinit, SNOWPACK, MELTWATER, SM, SUZ, SLZ = initmodel(xinit, parameters, mu, muwts, rtwts,
                                                                      bufftime=0, outstate=True, routOpt=False, comprout=False)
        else:

            # Without buff time, initialize state variables with zeros
            Ngrid = x.shape[1]
            SNOWPACK = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            MELTWATER = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            SM = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            SUZ = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            SLZ = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()

        P = x[bufftime:, :, 0]
        Pm= P.unsqueeze(2).repeat(1,1,mu)
        T = x[bufftime:, :, 1]
        Tm = T.unsqueeze(2).repeat(1,1,mu)
        ETpot = x[bufftime:, :, 2]
        ETpm = ETpot.unsqueeze(2).repeat(1,1,mu)


        ## scale the parameters
        parascaLst = [[1,6], [50,1000], [0.05,0.9], [0.01,0.5], [0.001,0.2], [0.2,1],
                        [0,10], [0,100], [-2.5,2.5], [0.5,10], [0,0.1], [0,0.2], [0.3,5],[0,1]]
        routscaLst = [[0,2.9], [0,6.5]]

        parBETA = parascaLst[0][0] + parameters[:,0,:]*(parascaLst[0][1]-parascaLst[0][0])
        parFC = parascaLst[1][0] + parameters[:,1,:]*(parascaLst[1][1]-parascaLst[1][0])
        parK0 = parascaLst[2][0] + parameters[:,2,:]*(parascaLst[2][1]-parascaLst[2][0])
        parK1 = parascaLst[3][0] + parameters[:,3,:]*(parascaLst[3][1]-parascaLst[3][0])
        parK2 = parascaLst[4][0] + parameters[:,4,:]*(parascaLst[4][1]-parascaLst[4][0])
        parLP = parascaLst[5][0] + parameters[:,5,:]*(parascaLst[5][1]-parascaLst[5][0])
        parPERC = parascaLst[6][0] + parameters[:,6,:]*(parascaLst[6][1]-parascaLst[6][0])
        parUZL = parascaLst[7][0] + parameters[:,7,:]*(parascaLst[7][1]-parascaLst[7][0])
        parTT = parascaLst[8][0] + parameters[:,8,:]*(parascaLst[8][1]-parascaLst[8][0])
        parCFMAX = parascaLst[9][0] + parameters[:,9,:]*(parascaLst[9][1]-parascaLst[9][0])
        parCFR = parascaLst[10][0] + parameters[:,10,:]*(parascaLst[10][1]-parascaLst[10][0])
        parCWH = parascaLst[11][0] + parameters[:,11,:]*(parascaLst[11][1]-parascaLst[11][0])

        # The added ET parameter
        parBETAET = parascaLst[12][0] + parameters[:,12,:]*(parascaLst[12][1]-parascaLst[12][0])
        parC = parascaLst[13][0] + parameters[:, 13] * (parascaLst[13][1] - parascaLst[13][0])
        Nstep, Ngrid = P.size()

        # Initialize time series of model variables
        Qsimmu = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()

        for t in range(Nstep):
            # Separate precipitation into liquid and solid components
            PRECIP = Pm[t, :, :]  # need to check later, seems repeating with line 52
            RAIN = torch.mul(PRECIP, (Tm[t, :, :] >= parTT).type(torch.float32))
            SNOW = torch.mul(PRECIP, (Tm[t, :, :] < parTT).type(torch.float32))

            # Snow
            SNOWPACK = SNOWPACK + SNOW
            melt = parCFMAX * (Tm[t, :, :] - parTT)
            # melt[melt < 0.0] = 0.0
            melt = torch.clamp(melt, min=0.0)
            # melt[melt > SNOWPACK] = SNOWPACK[melt > SNOWPACK]
            melt = torch.min(melt, SNOWPACK)
            MELTWATER = MELTWATER + melt
            SNOWPACK = SNOWPACK - melt
            refreezing = parCFR * parCFMAX * (parTT - Tm[t, :, :])
            # refreezing[refreezing < 0.0] = 0.0
            # refreezing[refreezing > MELTWATER] = MELTWATER[refreezing > MELTWATER]
            refreezing = torch.clamp(refreezing, min=0.0)
            refreezing = torch.min(refreezing, MELTWATER)
            SNOWPACK = SNOWPACK + refreezing
            MELTWATER = MELTWATER - refreezing
            tosoil = MELTWATER - (parCWH * SNOWPACK)
            # tosoil[tosoil < 0.0] = 0.0
            tosoil = torch.clamp(tosoil, min=0.0)
            MELTWATER = MELTWATER - tosoil

            # Soil and evaporation
            soil_wetness = (SM / parFC) ** parBETA
            # soil_wetness[soil_wetness < 0.0] = 0.0
            # soil_wetness[soil_wetness > 1.0] = 1.0
            soil_wetness = torch.clamp(soil_wetness, min=0.0, max=1.0)
            recharge = (RAIN + tosoil) * soil_wetness

            excess = SM - parFC
            # excess[excess < 0.0] = 0.0
            excess = torch.clamp(excess, min=0.0)
            SM = SM - excess
            # MODIFY HERE. Add the shape para parBETAET for ET equation
            evapfactor = (SM / (parLP * parFC)) ** parBETAET
            # evapfactor = SM / (parLP * parFC)
            # evapfactor[evapfactor < 0.0] = 0.0
            # evapfactor[evapfactor > 1.0] = 1.0
            evapfactor  = torch.clamp(evapfactor, min=0.0, max=1.0)
            ETact = ETpm[t, :, :] * evapfactor
            ETact = torch.min(SM, ETact)
            SM = torch.clamp(SM - ETact, min=PRECS) # SM can not be zero for gradient tracking

            SM = SM + RAIN + tosoil - recharge
            capillary = torch.min(SLZ, parC * SLZ * (1.0 - torch.clamp(SM / parFC, max=1.0)))
            
            SM = torch.clamp(SM + capillary, min=PRECS)
            SLZ = torch.clamp(SLZ - capillary, min=PRECS)

            # Groundwater boxes
            SUZ = SUZ + recharge + excess
            PERC = torch.min(SUZ, parPERC)
            SUZ = SUZ - PERC
            Q0 = parK0 * torch.clamp(SUZ - parUZL, min=0.0)
            SUZ = SUZ - Q0
            Q1 = parK1 * SUZ
            SUZ = SUZ - Q1
            SLZ = SLZ + PERC
            Q2 = parK2 * SLZ
            SLZ = SLZ - Q2
            Qsimmu[t, :, :] = Q0 + Q1 + Q2


        # get the initial average
        if muwts is None:
            Qsimave = Qsimmu.mean(-1)
        else:
            Qsimave = (Qsimmu * muwts).sum(-1)

        if routOpt is True: # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim = Qsimmu.view(Nstep, Ngrid * mu)
            else:
                # average the components, then do routing
                Qsim = Qsimave

            tempa = routscaLst[0][0] + rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb = routscaLst[1][0] + rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])
            routa = tempa.repeat(Nstep, 1).unsqueeze(-1)
            routb = tempb.repeat(Nstep, 1).unsqueeze(-1)
            UH = UH_gamma(routa, routb, lenF=15)  # lenF: folter
            rf = torch.unsqueeze(Qsim, -1).permute([1, 2, 0])   # dim:gage*var*time
            UH = UH.permute([1, 2, 0])  # dim: gage*var*time
            Qsrout = UH_conv(rf, UH).permute([2, 0, 1])

            if comprout is True: # Qs is [time, [gage*mult], var] now
                Qstemp = Qsrout.view(Nstep, Ngrid, mu)
                if muwts is None:
                    Qs = Qstemp.mean(-1, keepdim=True)
                else:
                    Qs = (Qstemp * muwts).sum(-1, keepdim=True)
            else:
                Qs = Qsrout

        else: # no routing, output the initial average simulations

            Qs = torch.unsqueeze(Qsimave, -1) # add a dimension

        if outstate is True:
            return Qs, SNOWPACK, MELTWATER, SM, SUZ, SLZ
        else:
            return Qs # total streamflow

