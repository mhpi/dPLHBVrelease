# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 11:40:57 2020

@author: axs6878
"""

# -*- coding: utf-8 -*-
"""
This script is used to read the streamflow data from GAGES II
"""

import os
import pandas as pd
import numpy as np
import datetime as dt
from datetime import date, timedelta
from hydroDL import utils, pathGAGES
from pandas.api.types import is_numeric_dtype, is_string_dtype
import time
import json
from . import Dataframe

# module variable
tRange = [19850101, 20151231]
start_date = '1990-1-1'
end_date = '2010-12-31'
tRangeobs = [19850101, 20151231]  # streamflow observations
tLst = utils.time.tRange2Array(tRange)
tLstobs = utils.time.tRange2Array(tRangeobs)
nt = len(tLst)
ntobs = len(tLstobs)

forcingLst = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
attrLstSel = [
    'ELEV_MEAN_M_BASIN', 'SLOPE_PCT', 'DRAIN_SQKM', 'HYDRO_DISTURB_INDX', 'STREAMS_KM_SQ_KM',
    'BFI_AVE', 'NDAMS_2009', 'STOR_NID_2009', 'RAW_DIS_NEAREST_DAM', 'FRAGUN_BASIN', 'DEVNLCD06',
    'FORESTNLCD06', 'PLANTNLCD06', 'AWCAVE', 'PERMAVE', 'RFACT', 'PPTAVG_BASIN'
]

LanduseAttrAll=[
 'NWALTXX_DEV_SUM', 'NWALTXX_SEMIDEV_SUM', 'NWALTXX_AG4344_SUM', 'NWALTXX_AG4346_SUM',
 'NWALTXX_11', 'NWALTXX_XX', 'NWALTXX_21', 'NWALTXX_22', 'NWALTXX_23', 'NWALTXX_24',
 'NWALTXX_25', 'NWALTXX_26', 'NWALTXX_27', 'NWALTXX_31', 'NWALTXX_32', 'NWALTXX_33',
 'NWALTXX_41', 'NWALTXX_43', 'NWALTXX_44', 'NWALTXX_45', 'NWALTXX_46', 'NWALTXX_50',
 'NWALTXX_60'
]

LanduseAttr = ['NWALTXX_DEV_SUM', 'NWALTXX_AG4346_SUM', 'NWALTXX_11', 'NWALTXX_25', 'NWALTXX_43',
                   'NWALTXX_50']

def readGageInfo(dirDB):
    gage_info_file_path = os.path.join(dirDB, "gage_info", "gage_info_complete.txt")
    data = pd.read_csv(gage_info_file_path, sep='\t')
    # fieldLst = ['huc', 'id', 'name', 'lat', 'lon', 'area']
    out = dict()
    # fieldLst = ['id', 'lat', 'lon', 'area', 'lat', 'lon', 'area']
    out = dict()
    out['id'] = data['STAID'].to_numpy()
    out['lat'] = data['LAT_GAGE'].to_numpy()
    out['lon'] = data['LNG_GAGE'].to_numpy()
    out['area'] = data['DRAIN_SQKM'].to_numpy()
    out['huc'] = data['HUC02'].to_numpy()
    out['HDI'] = data['HYDRO_DISTURB_INDX'].to_numpy()
    out['Class'] = data['CLASS']

    return out


def readUsgsGage(gageID, readQc=False):
    gage_data_file = os.path.join(dirDB, "gage_data", 'USGS_%08d.csv' % (gageID))
    dataTemp = pd.read_csv(gage_data_file, sep='\t')
    # print(gageID)
    date_time = dataTemp["datetime"]
    flow = dataTemp["00060_Mean"]
    flow_qc = dataTemp["00060_Mean_cd"]

    if len(flow) != ntobs:
        out = np.full([ntobs], np.nan)
        date = pd.to_datetime(date_time, format='%Y-%m-%d').values.astype('datetime64[ns]')
        [C, ind1, ind2] = np.intersect1d(date, tLstobs, return_indices=True)
        out1 = flow[ind1]
        out[ind2] = out1
    else:
        out = flow
        outQc = flow_qc

    if readQc is True:
        qcDict = {'A': 1, 'P': 2, 'A, e': 3, 'P, e': 4, 'P, Ice': 5}
        qc = np.array([qcDict[x] for x in flow_qc])

    if readQc is True:
        return out, outQc
    else:
        return out


def readUsgs(usgsIdLst):
    """
    This function reads data for all the USGS points listed in usgsIdLst
    :param usgsIdLst:
    :return: y
    """
    print("Reading GAGES data")
    t0 = time.time()
    y = np.empty([len(usgsIdLst), ntobs])
    for k in range(len(usgsIdLst)):
        dataObs = readUsgsGage(usgsIdLst[k])
        y[k, :] = dataObs
    y[y<0] = 0.0
    print("read usgs streamflow", time.time() - t0)
    return y


def readForcingGage(usgsId, varLst=forcingLst, *, dataset='daymet4'):
    # dataset = daymet or maurer or nldas
    forcingLst = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
    ind = np.argwhere(gageDict['id'] == usgsId)[0][0]
    huc = gageDict['huc'][ind]

    dataFolder = os.path.join(dirDB, 'forcing')

    # # if dataset is 'daymet':
    # #     tempS = 'cida'
    # # else:
    #     tempS = dataset

    tempS = 'daymet'

    # # Commenting for trial
    # if (huc < '16'):
    #     dataFile = os.path.join(dataFolder, dataset,
    #                             str(huc).zfill(2),
    #                             '08%d_lump_%s_forcing.txt' % (usgsId, tempS))
    # else:
    #     dataFile = os.path.join(dataFolder, dataset,
    #                             str(huc).zfill(2),
    #                             '08%d_lump_%s_forcing.txt' % (usgsId, tempS))

    dataFile = os.path.join(dataFolder, dataset,
                        str(huc).zfill(2),
                        '%08d_lump_%s_forcing.txt' % (usgsId, tempS))

    dataTemp = pd.read_csv(dataFile, sep=r'\s+')
    # dataTemp = pd.read_csv(dataFile)
    dfDate = dataTemp[['Year', 'Mnth', 'Day']]
    dfDate.columns = ['year', 'month', 'day']

    # datevalue = pd.to_datetime(dfDate).values.astype('datetime64[D]')
    # dataTemp.columns = range(11)

    dataTemp['date'] = pd.to_datetime(dfDate)

    begin_date = min(dataTemp['date'])
    end_date = max(dataTemp['date'])
    rng = pd.date_range(start=begin_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), freq="D")

    dataTemp.set_index(['date'], inplace=True)
    df = dataTemp.reindex(rng, method='pad')
    df = df.reset_index()
    mask = (df['index'].dt.day == 31) & (df['index'].dt.month == 12) & (df['index'].dt.is_leap_year)
    df.loc[mask, 'Day'] = 31
    # df.drop(columns=['index'])
    df.columns = range(12)

    [C, ind1, ind2] = np.intersect1d(rng, tLstobs, return_indices=True)

    # [C, ind1, ind2] = np.intersect1d(datevalue, tLstobs, return_indices=True)
    df_select = df.loc[ind1]

    nf = len(varLst)
    # out = np.empty([nt, nf])
    # for k in range(nf):
    #     # assume all files are of same columns. May check later.
    #     ind = forcingLst.index(varLst[k])
    #
    #     out[:, k] = df_select[ind + 5].values
    out = df_select.loc[:, 5:].values
    if out.shape[-1] != nf:
        raise Exception('Data read out not consistent with forcing varLst')
    return out


def readForcing(usgsIdLst, varLst):
    print("Reading forcing data")
    t0 = time.time()
    x = np.empty([len(usgsIdLst), nt, len(varLst)])
    for k in range(len(usgsIdLst)):
        data = readForcingGage(usgsIdLst[k], varLst)
        x[k, :, :] = data
    print("read forcing", time.time() - t0)
    return x

def readLanduse(usgsIdLst, varlst=None):
    print("Reading Land use data")
    t0= time.time()
    if varlst is None:
        LanduseAttr = ['NWALTXX_DEV_SUM', 'NWALTXX_AG4346_SUM', 'NWALTXX_11', 'NWALTXX_25', 'NWALTXX_43',
                   'NWALTXX_50']
    else:
        LanduseAttr = varlst

    dataFolder = os.path.join(dirDB, 'Dataset5_LandUse')
    years = [1974, 1982, 1992, 2002, 2012]

    data = np.ndarray(shape=(9067, 23, 5))  # [gages, attributes,years]

    for i in range(len(years)):
        year = years[i]
        filename = "LandUse_NWALT_" + str(year) + ".txt"
        data_year = pd.read_csv(os.path.join(dataFolder, filename))
        gageIdsLanduse = data_year['STAID'].to_numpy()
        data_year = data_year.set_index('STAID')
        data_numpy = data_year.to_numpy()
        data[:, :, i] = data_numpy

    C, ind1, ind2 = np.intersect1d(gageIdsLanduse, usgsIdLst, return_indices=True)
    data = data[ind1, :, :]

    both = set(LanduseAttr).intersection(LanduseAttrAll)
    indLanduse = [LanduseAttrAll.index(x) for x in both]
    data = data[:, indLanduse, :]

    rng = pd.date_range(start='01/01/1974', end='31/12/2015', freq="D")
    interpolated_data = np.ndarray(shape=(data.shape[0], data.shape[1], len(rng)))

    # Repeating the values of 2012 to 2015
    ndata = np.ndarray(shape=(data.shape[0], data.shape[1], data.shape[2] + 1))
    ndata[:, :, 0:data.shape[2]] = data
    ndata[:, :, data.shape[2]] = data[:, :, data.shape[2] - 1]


    for gage in range(interpolated_data.shape[0]):
        for attr in range(interpolated_data.shape[1]):
            dataseries = ndata[gage, attr, :]
            dates = ['01/01/1974', '01/01/1982', '01/01/1992', '01/01/2002', '31/12/2012', '31/12/2015']
            dates2 = pd.to_datetime(dates)
            data_dict = {'date': pd.to_datetime(dates), 'values': dataseries}

            df0 = pd.DataFrame(data_dict, columns=['date', 'values'])

            df = df0.copy()
            df['date'] = pd.to_datetime(df['date'])
            df.index = df['date']
            del df['date']


            df_interpol = df.resample('D').mean()
            df_interpol['values'] = df_interpol['values'].interpolate()
            ts = df_interpol.to_numpy().reshape(len(rng))
            interpolated_data[gage, attr, :] = ts

    interpolated_data = np.swapaxes(interpolated_data, 1, 2)

    # # Repeating the values of 31/12/2012 for next three years
    # for i in range(len(rng1)):
    #     a = interpolated_data[:, -1, :]
    #     a1 = a.reshape([a.shape[0], 1, a.shape[1]])
    #     interpolated_data = np.append(interpolated_data, a1, axis=1)

    rng = pd.date_range(start='01/01/1974', end='31/12/2015', freq="D")
    [C, ind1, ind2] = np.intersect1d(rng, tLstobs, return_indices=True)
    interpolated_data = interpolated_data[:, ind1, :]

    print("read landuse", time.time() - t0)
    return interpolated_data

def readWateruse(usgsIdLst):
    print("Reading Water use data")
    t0 = time.time()

    dataFolder = os.path.join(dirDB, 'Dataset10_WaterUse', 'Dataset10_WaterUse')
    file = os.path.join(dataFolder, 'WaterUse_1985-2010.txt')
    df = pd.read_csv(file)

    data = df.to_numpy()
    gageIdsWateruse = data[:, 0]
    data = data[:, 1:]
    data = np.column_stack((data, data[:, 5]))

    C, ind1, ind2 = np.intersect1d(gageIdsWateruse, usgsIdLst, return_indices=True)
    data = data[ind1, :]

    interpolated_data = np.ndarray(shape=(data.shape[0], 11322))

    for gage in range(data.shape[0]):
        dataseries = data[gage, :]
        dates = ['01/01/1985', '01/01/1990', '01/01/1995', '01/01/2000',
                 '01/01/2005', '01/01/2010', '12/31/2015']

        data_dict = {'date': pd.to_datetime(dates), 'values': dataseries}

        df0 = pd.DataFrame(data_dict, columns=['date', 'values'])

        df = df0.copy()
        df['date'] = pd.to_datetime(df['date'])
        df.index = df['date']
        del df['date']

        df_interpol = df.resample('D').mean()
        df_interpol['values'] = df_interpol['values'].interpolate()
        ts = df_interpol.to_numpy().reshape(11322)
        interpolated_data[gage, :] = ts

    rng = pd.date_range(start='01/01/1985', end='31/12/2015', freq="D")
    [C, ind1, ind2] = np.intersect1d(rng, tLstobs, return_indices=True)
    interpolated_data = interpolated_data[:, ind1]

    print("read wateruse", time.time() - t0)
    return interpolated_data


def readAttrAll(idLst, *, saveDict=False):
    print("Reading Attribute data")
    t0 = time.time()
    attrFile = os.path.join(dirDB, 'attributes', 'attributes.txt')
    data = pd.read_csv(attrFile, sep="\t")
    # This file as attributes for all stations.
    # Therefore, selecting the gages used in this study only.
    gageIds = data['STAID'].to_numpy()
    [C, ind1, ind2] = np.intersect1d(idLst, gageIds, return_indices=True)
    data_select = data.iloc[ind2]

    varLst = data_select.columns.values.tolist()
    varLst = varLst[1:]  # Skipping the first column (Station Id)
    data_select = data_select.drop(columns=['STAID'])
    out = data_select.to_numpy()
    print("read Attributes data", time.time() - t0)
    return out, varLst


def readAttr(usgsIdLst, varLst):

    attrAll, varLstAll = readAttrAll(usgsIdLst)
    indVar = list()
    for var in varLst:
        indVar.append(varLstAll.index(var))
    idLstAll = gageDict['id']
    C, indGrid, ind2 = np.intersect1d(idLstAll, usgsIdLst, return_indices=True)
    temp = attrAll[indGrid, :]
    out = temp[:, indVar]
    return out


def calStat(x):
    a = x.flatten()
    b = a[~np.isnan(a)]  # kick out Nan
    p10 = np.percentile(b, 10).astype(float)
    p90 = np.percentile(b, 90).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)
    if std < 0.001:
        std = 1
    return [p10, p90, mean, std]


def calStatgamma(x):  # for daily streamflow and precipitation
    a = x.flatten()
    b = a[~np.isnan(a)]  # kick out Nan
    b = np.log10(np.sqrt(b) + 0.1)  # do some tranformation to change gamma characteristics
    p10 = np.percentile(b, 10).astype(float)
    p90 = np.percentile(b, 90).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)
    if std < 0.001:
        std = 1
    return [p10, p90, mean, std]


def calStatbasinnorm(x):  # for daily streamflow normalized by basin area and precipitation
    basinarea = readAttr(gageDict['id'], ['DRAIN_SQKM'])
    meanprep = readAttr(gageDict['id'], ['PPTAVG_BASIN'])
    # meanprep = readAttr(gageDict['id'], ['q_mean'])
    temparea = np.tile(basinarea, (1, x.shape[1]))
    tempprep = np.tile(meanprep, (1, x.shape[1]))
    flowua = (x * 0.0283168 * 3600 * 24) / (
                (temparea * (10 ** 6)) * (tempprep * 10 ** (-2)))  # unit (m^3/day)/(m^3/day)
    a = flowua.flatten()
    b = a[~np.isnan(a)]  # kick out Nan
    b = np.log10(np.sqrt(b) + 0.1)  # do some tranformation to change gamma characteristics plus 0.1 for 0 values
    p10 = np.percentile(b, 10).astype(float)
    p90 = np.percentile(b, 90).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)
    if std < 0.001:
        std = 1
    return [p10, p90, mean, std]


def calStatAll():
    statDict = dict()
    idLst = gageDict['id']
    # usgs streamflow
    y = readUsgs(idLst)
    # statDict['usgsFlow'] = calStatgamma(y)
    statDict['usgsFlow'] = calStatbasinnorm(y)
    # forcing
    x = readForcing(idLst, forcingLst)
    for k in range(len(forcingLst)):
        var = forcingLst[k]
        if var == 'prcp':
            statDict[var] = calStatgamma(x[:, :, k])
        else:
            statDict[var] = calStat(x[:, :, k])
    # const attribute
    attrData, attrLst = readAttrAll(idLst)
    for k in range(len(attrLst)):
        var = attrLst[k]
        statDict[var] = calStat(attrData[:, k])
    statFile = os.path.join(dirDB, 'Statistics_basinnorm1.json')
    with open(statFile, 'w') as fp:
        json.dump(statDict, fp, indent=4)


def transNorm(x, varLst, *, toNorm):
    if type(varLst) is str:
        varLst = [varLst]
    out = np.zeros(x.shape)
    for k in range(len(varLst)):
        var = varLst[k]
        stat = statDict[var]
        if toNorm is True:
            if len(x.shape) == 3:
                if var == 'prcp' or var == 'usgsFlow':
                    x[:, :, k] = np.log10(np.sqrt(x[:, :, k]) + 0.1)
                out[:, :, k] = (x[:, :, k] - stat[2]) / stat[3]
            elif len(x.shape) == 2:
                if var == 'prcp' or var == 'usgsFlow':
                    x[:, k] = np.log10(np.sqrt(x[:, k]) + 0.1)
                out[:, k] = (x[:, k] - stat[2]) / stat[3]
        else:
            if len(x.shape) == 3:
                out[:, :, k] = x[:, :, k] * stat[3] + stat[2]
                if var == 'prcp' or var == 'usgsFlow':
                    out[:, :, k] = (np.power(10, out[:, :, k]) - 0.1) ** 2
            elif len(x.shape) == 2:
                out[:, k] = x[:, k] * stat[3] + stat[2]
                if var == 'prcp' or var == 'usgsFlow':
                    out[:, k] = (np.power(10, out[:, k]) - 0.1) ** 2
    return out


def basinNorm(x, gageid, toNorm):
    # for regional training, gageid should be numpyarray
    if type(gageid) is str:
        if gageid == 'All':
            gageid = gageDict['id']
    nd = len(x.shape)
    basinarea = readAttr(gageid, ['DRAIN_SQKM'])
    meanprep = readAttr(gageid, ['PPTAVG_BASIN'])
    # meanprep = readAttr(gageid, ['q_mean'])
    if nd == 3 and x.shape[2] == 1:
        x = x[:, :, 0]  # unsqueeze the original 3 dimension matrix
    temparea = np.tile(basinarea, (1, x.shape[1]))
    tempprep = np.tile(meanprep, (1, x.shape[1]))
    if toNorm is True:
        flow = (x * 0.0283168 * 3600 * 24) / ((temparea * (10 ** 6)) * (tempprep * 10 ** (-2)))  # (m^3/day)/(m^3/day)
    else:

        flow = x * ((temparea * (10 ** 6)) * (tempprep * 10 ** (-2))) / (0.0283168 * 3600 * 24)
    if nd == 3:
        flow = np.expand_dims(flow, axis=2)
    return flow


def createSubsetAll(opt, **kw):
    if opt is 'all':
        idLst = gageDict['id']
        subsetFile = os.path.join(dirDB, 'Subset', 'all.csv')
        np.savetxt(subsetFile, idLst, delimiter=',', fmt='%d')


# Define and initialize module variables
if os.path.isdir(pathGAGES['DB']):
    dirDB = pathGAGES['DB']
    gageDict = readGageInfo(dirDB)
    statFile = os.path.join(dirDB, 'Statistics_basinnorm1.json')
    if not os.path.isfile(statFile):
        calStatAll()
    with open(statFile, 'r') as fp:
        statDict = json.load(fp)
else:
    dirDB = None
    gageDict = None
    statDict = None


def initgages(rootDB=pathGAGES['DB']):
    # reinitialize module variable
    global dirDB, gageDict, statDict
    dirDB = rootDB
    gageDict = readGageInfo(dirDB)
    statFile = os.path.join(dirDB, 'Statistics_basinnorm1.json')
    if not os.path.isfile(statFile):
        calStatAll()
    with open(statFile, 'r') as fp:
        statDict = json.load(fp)


class DataframeGages(Dataframe):
    def __init__(self, *, subset='All', tRange):
        self.subset = subset
        if subset == 'All':  # change to read subset later
            self.usgsId = gageDict['id']
            crd = np.zeros([len(self.usgsId), 2])
            crd[:, 0] = gageDict['lat']
            crd[:, 1] = gageDict['lon']
            self.crd = crd
        elif type(subset) is list:
            self.usgsId = np.array(subset)
            crd = np.zeros([len(self.usgsId), 2])
            C, ind1, ind2 = np.intersect1d(self.usgsId, gageDict['id'], return_indices=True)
            crd[:, 0] = gageDict['lat'][ind2]
            crd[:, 1] = gageDict['lon'][ind2]
            self.crd = crd
        else:
            raise Exception('The format of subset is not correct!')
        self.time = utils.time.tRange2Array(tRange)

    def getGeo(self):
        return self.crd

    def getT(self):
        return self.time

    def getDataObs(self, *, doNorm=True, rmNan=True, basinnorm=True):
        data = readUsgs(self.usgsId)
        if basinnorm is True:
            data = basinNorm(data, gageid=self.usgsId, toNorm=True)
        data = np.expand_dims(data, axis=2)
        C, ind1, ind2 = np.intersect1d(self.time, tLstobs, return_indices=True)
        data = data[:, ind2, :]
        if doNorm is True:
            data = transNorm(data, 'usgsFlow', toNorm=True)
        if rmNan is True:
            data[np.where(np.isnan(data))] = 0
            # data[np.where(np.isnan(data))] = -99
        return data

    def getDataTs(self, *, varLst=forcingLst, doNorm=True, rmNan=True,includeLanduse,includeWateruse):
        if type(varLst) is str:
            varLst = [varLst]
        # read ts forcing
        dataForcing = readForcing(self.usgsId, varLst)  # data:[gage*day*variable]
        C, ind1, ind2 = np.intersect1d(self.time, tLst, return_indices=True)
        dataForcing = dataForcing[:, ind2, :]
        data = dataForcing

        # read Landuse
        if includeLanduse is True:
            dataLanduse = readLanduse(self.usgsId, LanduseAttr)
            C, ind1, ind2 = np.intersect1d(self.time, tLst, return_indices=True)
            dataLanduse = dataLanduse[:, ind2, :]
            data = np.concatenate((data, dataLanduse), axis=2)

        # read Wateruse
        if includeWateruse is True:
            dataWateruse = readWateruse(self.usgsId)
            C, ind1, ind2 = np.intersect1d(self.time, tLst, return_indices=True)
            dataWateruse = dataWateruse[:, ind2]
            dataWateruse = dataWateruse.reshape([dataWateruse.shape[0], dataWateruse.shape[1], 1])
            data = np.concatenate((data, dataWateruse), axis=2)


        if doNorm is True:
            data = transNorm(data, varLst, toNorm=True)
        if rmNan is True:
            data[np.where(np.isnan(data))] = 0
        return data

    def getDataConst(self, *, varLst=attrLstSel, doNorm=True, rmNan=True):
        if type(varLst) is str:
            varLst = [varLst]
        data = readAttr(self.usgsId, varLst)
        if doNorm is True:
            data = transNorm(data, varLst, toNorm=True)
        if rmNan is True:
            data[np.where(np.isnan(data))] = 0
        return data