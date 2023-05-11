import sys
sys.path.append('../../')
from hydroDL import master, utils
from hydroDL.data import camels
from hydroDL.master import loadModel
from hydroDL.model import train
from hydroDL.post import plot, stat

import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import random
import json
import datetime as dt


## fix the random seeds
randomseed = 111111
random.seed(randomseed)
torch.manual_seed(randomseed)
np.random.seed(randomseed)
torch.cuda.manual_seed(randomseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

## GPU setting
testgpuid = 0
torch.cuda.set_device(testgpuid)

## setting options, keep the same as your training
PUOpt = 0  # 0 for All; 1 for PUB; 2 for PUR;
buffOptOri = 0  # original buffOpt, must be same as what you set for training
buffOpt = 0  # control load training data 0: do nothing; 1: repeat first year; 2: load one more year
forType = 'daymet'

## Hyperparameters, keep the same as your training setup
BATCH_SIZE = 100
RHO = 365
HIDDENSIZE = 256
Ttrain = [19801001, 19951001]  # Training period
# Ttrain = [19891001, 19991001]  # PUB/PUR period
Tinv = [19801001, 19951001] # dPL Inversion period
# Tinv = [19891001, 19991001]  # PUB/PUR period
Nfea = 12 # number of HBV parameters
BUFFTIME = 365
routing = True
Nmul = 16
comprout = False
compwts = False
pcorr = None

## Testing parameters
Ttest = [19951001, 20101001]  # testing period
# Ttest = [19891001, 19991001]  # PUB/PUR period
TtestLst = utils.time.tRange2Array(Ttest)
TtestLoad = [19951001, 20101001]  # could potentially use this to load more forcings before testing period as warm-up
# TtestLoad = [19801001, 19991001]  # PUB/PUR period
testbatch = 50  # forward number of "testbatch" basins each time to save GPU memory. You can set this even smaller to save more.
testepoch = 50

testseed = 111111

# Define root directory of database and saved output dir
# Modify this based on your own location of CAMELS dataset and saved models
rootDatabase = os.path.join(os.path.sep, 'scratch', 'Camels')  # CAMELS dataset root directory
camels.initcamels(rootDatabase)  # initialize three camels module-scope variables in camels.py: dirDB, gageDict, statDict

rootOut = os.path.join(os.path.sep, 'data', 'rnnStreamflow')  # Model output root directory

# CAMLES basin info
gageinfo = camels.gageDict
hucinfo = gageinfo['huc']
gageid = gageinfo['id']
gageidLst = gageid.tolist()

# same as training, load data based on ALL, PUB, PUR scenarios
if PUOpt == 0: # for All the basins
    puN = 'ALL'
    tarIDLst = [gageidLst]

elif PUOpt == 1: # for PUB
    puN = 'PUB'
    # load the subset ID
    # splitPath saves the basin ID of random groups
    splitPath = 'PUBsplitLst.txt'
    with open(splitPath, 'r') as fp:
        testIDLst=json.load(fp)
    tarIDLst = testIDLst

elif PUOpt == 2: # for PUR
    puN = 'PUR'
    # Divide CAMELS dataset into 7 PUR regions
    # get the id list of each region
    regionID = list()
    regionNum = list()
    regionDivide = [ [1,2], [3,6], [4,5,7], [9,10], [8,11,12,13], [14,15,16,18], [17] ] # seven regions
    for ii in range(len(regionDivide)):
        tempcomb = regionDivide[ii]
        tempregid = list()
        for ih in tempcomb:
            tempid = gageid[hucinfo==ih].tolist()
            tempregid = tempregid + tempid
        regionID.append(tempregid)
        regionNum.append(len(tempregid))
    tarIDLst = regionID

# define the matrix to save results
predtestALL = np.full([len(gageid), len(TtestLst), 5], np.nan)
obstestALL = np.full([len(gageid), len(TtestLst), 1], np.nan)

# this testsave_path should be consistent with where you save your model
testsave_path = 'CAMELSDemo/dPLHBV/' + puN + '/Testforc/' + forType + '/BuffOpt' + str(buffOptOri) +\
                '/RMSE_para0.25/'+str(testseed)

## load data and test the model
nstart = 0
logtestIDLst = []
# loop to test all the folds for PUB and PUR. The default is you need to have run all folds, but if
# you only run one fold for PUB or PUR and just want to test that fold (i.e. fold X), you may set this as:
# for ifold in range(X, X+1):
for ifold in range(1, len(tarIDLst)+1):
    testfold = ifold
    TestLS = tarIDLst[testfold - 1]
    TestInd = [gageidLst.index(j) for j in TestLS]
    if PUOpt == 0:  # Train and test on ALL basins
        TrainLS = gageidLst
        TrainInd = [gageidLst.index(j) for j in TrainLS]
    else:
        TrainLS = list(set(gageid.tolist()) - set(TestLS))
        TrainInd = [gageidLst.index(j) for j in TrainLS]
    gageDic = {'TrainID':TrainLS, 'TestID':TestLS}

    nbasin = len(TestLS) # number of basins for testing

    # get the dir path of the saved model for testing
    foldstr = 'Fold' + str(testfold)
    exp_info = 'T_'+str(Ttrain[0])+'_'+str(Ttrain[1])+'_BS_'+str(BATCH_SIZE)+'_HS_'+str(HIDDENSIZE)\
               +'_RHO_'+str(RHO)+'_NF_'+str(Nfea)+'_Buff_'+str(BUFFTIME)+'_Mul_'+str(Nmul)
    # the final path to test with the trained model saved in
    testout = os.path.join(rootOut, testsave_path, foldstr, exp_info)
    testmodel = loadModel(testout, epoch=testepoch)

    # apply buffOpt for loading the training data
    if buffOpt == 2: # load more "BUFFTIME" forcing before the training period
        sd = utils.time.t2dt(Ttrain[0]) - dt.timedelta(days=BUFFTIME)
        sdint = int(sd.strftime("%Y%m%d"))
        TtrainLoad = [sdint, Ttrain[1]]
        TinvLoad = [sdint, Ttrain[1]]
    else:
        TtrainLoad = Ttrain
        TinvLoad = Tinv

    # prepare input data
    # load camels dataset
    if forType is 'daymet':
        varF = ['prcp', 'tmean']
        varFInv = ['prcp', 'tmean']
    else:
        varF = ['prcp', 'tmax']  # tmax is tmean here for the original CAMELS maurer and nldas forcing
        varFInv = ['prcp', 'tmax']

    attrnewLst = [ 'p_mean','pet_mean','p_seasonality','frac_snow','aridity','high_prec_freq','high_prec_dur',
                   'low_prec_freq','low_prec_dur', 'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max',
                   'lai_diff', 'gvf_max', 'gvf_diff', 'dom_land_cover_frac', 'dom_land_cover', 'root_depth_50',
                   'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
                   'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 'geol_1st_class', 'glim_1st_class_frac',
                   'geol_2nd_class', 'glim_2nd_class_frac', 'carbonate_rocks_frac', 'geol_porostiy', 'geol_permeability']


    # for HBV training inputs
    dfTrain = camels.DataframeCamels(tRange=TtrainLoad, subset=TrainLS, forType=forType)
    forcUN = dfTrain.getDataTs(varLst=varF, doNorm=False, rmNan=False)
    # obsUN = dfTrain.getDataObs(doNorm=False, rmNan=False, basinnorm=False)  # useless for testing

    # for dPL inversion training data
    dfInv = camels.DataframeCamels(tRange=TinvLoad, subset=TrainLS, forType=forType)
    forcInvUN = dfInv.getDataTs(varLst=varFInv, doNorm=False, rmNan=False)
    attrsUN = dfInv.getDataConst(varLst=attrnewLst, doNorm=False, rmNan=False)

    # for HBV testing input
    dfTest = camels.DataframeCamels(tRange=TtestLoad, subset=TestLS, forType=forType)
    forcTestUN = dfTest.getDataTs(varLst=varF, doNorm=False, rmNan=False)
    obsTestUN = dfTest.getDataObs(doNorm=False, rmNan=False, basinnorm=False)
    attrsTestUN = dfTest.getDataConst(varLst=attrnewLst, doNorm=False, rmNan=False)

    # Transform obs from ft3/s to mm/day
    # areas = gageinfo['area'][TrainInd] # unit km2
    # temparea = np.tile(areas[:, None, None], (1, obsUN.shape[1],1))
    # obsUN = (obsUN * 0.0283168 * 3600 * 24) / (temparea * (10 ** 6)) * 10**3  # useless for testing
    areas = gageinfo['area'][TestInd] # unit km2
    temparea = np.tile(areas[:, None, None], (1, obsTestUN.shape[1],1))
    obsTestUN = (obsTestUN * 0.0283168 * 3600 * 24) / (temparea * (10 ** 6)) * 10**3 # transform to mm/day

    # load potential ET calculated by hargreaves method
    varLstNL = ['PEVAP']
    usgsIdLst = gageid
    if forType == 'maurer':
        tPETRange = [19800101, 20090101]
    else:
        tPETRange = [19800101, 20150101]
    tPETLst = utils.time.tRange2Array(tPETRange)
    PETDir = rootDatabase + '/pet_harg/' + forType + '/'
    ntime = len(tPETLst)
    PETfull = np.empty([len(usgsIdLst), ntime, len(varLstNL)])
    for k in range(len(usgsIdLst)):
        dataTemp = camels.readcsvGage(PETDir, usgsIdLst[k], varLstNL, ntime)
        PETfull[k, :, :] = dataTemp

    TtrainLst = utils.time.tRange2Array(TtrainLoad)
    TinvLst = utils.time.tRange2Array(TinvLoad)
    TtestLoadLst = utils.time.tRange2Array(TtestLoad)
    C, ind1, ind2 = np.intersect1d(TtrainLst, tPETLst, return_indices=True)
    PETUN = PETfull[:, ind2, :]
    PETUN = PETUN[TrainInd, :, :] # select basins
    C, ind1, ind2inv = np.intersect1d(TinvLst, tPETLst, return_indices=True)
    PETInvUN = PETfull[:, ind2inv, :]
    PETInvUN = PETInvUN[TrainInd, :, :]
    C, ind1, ind2test = np.intersect1d(TtestLoadLst, tPETLst, return_indices=True)
    PETTestUN = PETfull[:, ind2test, :]
    PETTestUN = PETTestUN[TestInd, :, :]

    # process data, do normalization and remove nan
    series_inv = np.concatenate([forcInvUN, PETInvUN], axis=2)
    seriesvarLst = varFInv + ['pet']
    # load the saved statistics
    statFile = os.path.join(testout, 'statDict.json')
    with open(statFile, 'r') as fp:
        statDict = json.load(fp)

    # normalize
    attr_norm = camels.transNormbyDic(attrsUN, attrnewLst, statDict, toNorm=True)
    attr_norm[np.isnan(attr_norm)] = 0.0
    series_norm = camels.transNormbyDic(series_inv, seriesvarLst, statDict, toNorm=True)
    series_norm[np.isnan(series_norm)] = 0.0

    attrtest_norm = camels.transNormbyDic(attrsTestUN, attrnewLst, statDict, toNorm=True)
    attrtest_norm[np.isnan(attrtest_norm)] = 0.0
    seriestest_inv = np.concatenate([forcTestUN, PETTestUN], axis=2)
    seriestest_norm = camels.transNormbyDic(seriestest_inv, seriesvarLst, statDict, toNorm=True)
    seriestest_norm[np.isnan(seriestest_norm)] = 0.0

    # prepare the inputs
    zTrain = series_norm
    xTrain = np.concatenate([forcUN, PETUN], axis=2) # HBV forcing
    xTrain[np.isnan(xTrain)] = 0.0

    if buffOpt == 1: # repeat the first year for buff
        zTrainIn = np.concatenate([zTrain[:,0:BUFFTIME,:], zTrain], axis=1)
        xTrainIn = np.concatenate([xTrain[:,0:BUFFTIME,:], xTrain], axis=1) # Bufftime for the first year
        # yTrainIn = np.concatenate([obsUN[:,0:BUFFTIME,:], obsUN], axis=1)
    else: # no repeat, original data
        zTrainIn = zTrain
        xTrainIn = xTrain
        # yTrainIn = obsUN

    forcTuple = (xTrainIn, zTrainIn)
    attrs = attr_norm

    ## Prepare the testing data and forward the trained model for testing
    # TestBuff = 365 # Use 365 days forcing to warm up the model for testing
    TestBuff = xTrain.shape[1]  # Use the whole training period to warm up the model for testing
    # TestBuff = len(TtestLoadLst) - len(TtestLst)  # use the redundantly loaded data to warm up

    # prepare file name to save the testing predictions
    filePathLst = master.master.namePred(
          testout, Ttest, 'All_Buff'+str(TestBuff), epoch=testepoch, targLst=['Qr', 'Q0', 'Q1', 'Q2', 'ET'])

    # prepare the inputs for TESTING
    if PUOpt == 0: # for ALL basins, temporal generalization test
        zTest = series_norm  # dPL inversion
        xTest = np.concatenate([forcTestUN, PETTestUN], axis=2)  # HBV forcing
        # forcings to warm up the model. Here use the forcing of training period to warm up
        xTestBuff = xTrain[:, -TestBuff:, :]
        xTest = np.concatenate([xTestBuff, xTest], axis=1)
        obs = obsTestUN[:, 0:, :]  # starts with 0 when not loading more data before testing period

    else:  # for PUB and PUR cases, different testing basins. Load more forcings to warm up.
        zTest = seriestest_norm[:, 0:TestBuff, :]  # Use the warm-up period forcing as the gA input in zTest
        # zTest = seriestest_norm
        xTest = np.concatenate([forcTestUN, PETTestUN], axis=2)  # HBV forcing
        obs = obsTestUN[:, TestBuff:, :]  # exclude loaded obs in warming up period (first TestBuff days) for evaluation

    # Use days of TestBuff to initialize the model
    testmodel.inittime=TestBuff

    # Final inputs to the test model
    xTest[np.isnan(xTest)] = 0.0
    attrtest = attrtest_norm
    cTemp = np.repeat(
        np.reshape(attrtest, [attrtest.shape[0], 1, attrtest.shape[-1]]), zTest.shape[1], axis=1)
    zTest = np.concatenate([zTest, cTemp], 2) # Add attributes to historical forcings as the inversion part
    testTuple = (xTest, zTest) # xTest: input forcings to HBV; zTest: inputs to gA LSTM to learn parameters

    # forward the model and save results
    train.testModel(
        testmodel, testTuple, c=None, batchSize=testbatch, filePathLst=filePathLst)

    # read out the saved forward predictions
    dataPred = np.ndarray([obs.shape[0], obs.shape[1], len(filePathLst)])
    for k in range(len(filePathLst)):
        filePath = filePathLst[k]
        dataPred[:, :, k] = pd.read_csv(
            filePath, dtype=np.float, header=None).values
    # save the predictions to the big matrix
    predtestALL[nstart:nstart+nbasin, :, :] = dataPred
    obstestALL[nstart:nstart+nbasin, :, :] = obs
    nstart = nstart + nbasin
    logtestIDLst = logtestIDLst + TestLS

## post processing
# calculate evaluation metrics
evaDict = [stat.statError(predtestALL[:,:,0], obstestALL.squeeze())]  # Q0: the streamflow

# save testing results
# Note: for PUB/PUR testing, this path location saves results covering all folds. If you are only testing one fold,
# consider add fold specification in "seStr" to prevent results overwritten from different folds.
seStr = 'Train'+str(Ttrain[0])+'_'+str(Ttrain[1])+'Test'+str(Ttest[0])+'_'+str(Ttest[1])+'Buff'+str(TestBuff)+'Nmul'+str(Nmul)
outpath = os.path.join(rootOut, testsave_path, seStr)
if not os.path.isdir(outpath):
    os.makedirs(outpath)

EvaFile = os.path.join(outpath, 'Eva'+str(testepoch)+'.npy')
np.save(EvaFile, evaDict)

obsFile = os.path.join(outpath, 'obs.npy')
np.save(obsFile, obstestALL)

predFile = os.path.join(outpath, 'pred'+str(testepoch)+'.npy')
np.save(predFile, predtestALL)

# calculate metrics for the widely used CAMELS subset with 531 basins
# we report on this 531 subset in Feng et al., 2022 HESSD
subsetPath = 'Sub531ID.txt'
with open(subsetPath, 'r') as fp:
    sub531IDLst = json.load(fp)  # Subset 531 ID List
# get the evaluation metrics on 531 subset
[C, ind1, SubInd] = np.intersect1d(sub531IDLst, logtestIDLst, return_indices=True)
evaframe = pd.DataFrame(evaDict[0])
evaframeSub = evaframe.loc[SubInd, list(evaframe.keys())]
evaS531Dict = [{col:evaframeSub[col].values for col in evaframeSub}] # 531 subset evaDict

# print NSE median value of testing basins
print('Testing finished! Evaluation results saved in\n', outpath)
print('For basins of whole CAMELS, NSE median:', np.nanmedian(evaDict[0]['NSE']))
print('For basins of 531 subset, NSE median:', np.nanmedian(evaS531Dict[0]['NSE']))

## Show boxplots of the results
evaDictLst = evaDict + evaS531Dict
plt.rcParams['font.size'] = 14
plt.rcParams["legend.columnspacing"] = 0.1
plt.rcParams["legend.handletextpad"] = 0.2
keyLst = ['NSE', 'KGE']
dataBox = list()
for iS in range(len(keyLst)):
    statStr = keyLst[iS]
    temp = list()
    for k in range(len(evaDictLst)):
        data = evaDictLst[k][statStr]
        data = data[~np.isnan(data)]
        temp.append(data)
    dataBox.append(temp)

labelname = ['dPL+HBV_Multi', 'dPL+HBV_Multi Sub531']
xlabel = keyLst
fig = plot.plotBoxFig(dataBox, xlabel, labelname, sharey=False, figsize=(6, 5))
fig.patch.set_facecolor('white')
fig.show()
plt.savefig(os.path.join(outpath, 'Metric_BoxPlot.png'), format='png')
