# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 04:26:48 2023

@author: Andriu
"""

#%% IMPORT LIBRARIES

import argparse
import numpy as np
import pandas as pd
import compress_pickle as cpickle

#%% SCRIPT ARGUMENTS

parser = argparse.ArgumentParser(prog= 'LPC_Dataset_Compare',
                                 description = 'Compares the analysis for the datasets...',
                                 epilog = 'Created by Andriu')

parser.add_argument('-d1','--dataset1', required=True, type=bool)
parser.add_argument('-d2','--dataset2', required=True, type=bool)
parser.add_argument('-d3','--dataset3', required=True, type=bool)

args = parser.parse_args()

# ARGUMENTS
d1 = args.dataset1
d2 = args.dataset2
d3 = args.dataset3

dsets = []
dates = []

if d1:
    dsets.append(1)
    dates.append({'s':'2010-09-08', 'e':'2016-08-22'})
if d2:
    dsets.append(2)
    dates.append({'s':'2016-08-23', 'e':'2022-08-08'})
if d3:
    dsets.append(3)
    dates.append({'s':'2010-09-08', 'e':'2022-08-08'})
    
#%% SCRIPT SETUP

infolder = 'output/analysis/'
ofolder = 'output/compare/'

dsets_paths = ['set'+str(x)+'/' for x in dsets]

#%% LOAD DATA

dfs = []

for i in range(len(dsets)):
    with open(infolder+dsets_paths[i]+'dfs.lzma','rb') as file:
        df = cpickle.load(file)
        dfs.append(df)

del(df, file, i)
    

dfrs_da = []
for i in range(len(dsets)):
    m = 'da'

    df = dfs[i][m]
    df.drop('normalized',axis=1,inplace=True)
    df.columns = [x.upper() for x in list(df.columns)]
    
    dfmax = df.max(axis=0)
    dfmin = df.min(axis=0)
    dfmean = df.mean(axis=0)
    dfstd = df.std(axis=0)
    dfcv = dfstd/dfmean
    
    dfc = pd.concat([dfmax, dfmin, dfmean, dfcv],axis=1)
    dfc.columns = ['MAX','MIN','MEAN','CV'] 
    
    dfm = pd.DataFrame([['dset'+str(i+1),np.nan,np.nan,np.nan],[dates[i]['s'],np.nan,np.nan,np.nan], [dates[i]['e'],np.nan,np.nan,np.nan]])
    dfm.index = ['dataset','start','end']
    dfm.columns = ['MAX','MIN','MEAN','CV'] 
    
    dfr = pd.concat([dfm, dfc],axis=0)
    
    dfrs_da.append(dfr)


dfrs_mae = []
for i in range(len(dsets)):
    m = 'mae'

    df = dfs[i][m]
    df.drop('normalized',axis=1,inplace=True)
    df.columns = [x.upper() for x in list(df.columns)]
    
    dfmax = df.max(axis=0)
    dfmin = df.min(axis=0)
    dfmean = df.mean(axis=0)
    dfstd = df.std(axis=0)
    dfcv = dfstd/dfmean
    
    dfc = pd.concat([dfmax, dfmin, dfmean, dfcv],axis=1)
    dfc.columns = ['MAX','MIN','MEAN','CV'] 
    
    dfm = pd.DataFrame([['dset'+str(i+1),np.nan,np.nan,np.nan],[dates[i]['s'],np.nan,np.nan,np.nan], [dates[i]['e'],np.nan,np.nan,np.nan]])
    dfm.index = ['dataset','start','end']
    dfm.columns = ['MAX','MIN','MEAN','CV'] 
    
    dfr = pd.concat([dfm, dfc],axis=0)
    
    dfrs_mae.append(dfr)

dfrs_dis = []
for i in range(len(dsets)):
    m = 'dis'

    df = dfs[i][m]
    df.drop('normalized',axis=1,inplace=True)
    df.columns = [x.upper() for x in list(df.columns)]
    
    dfmax = df.max(axis=0)
    dfmin = df.min(axis=0)
    dfmean = df.mean(axis=0)
    dfstd = df.std(axis=0)
    dfcv = dfstd/dfmean
    
    dfc = pd.concat([dfmax, dfmin, dfmean, dfcv],axis=1)
    dfc.columns = ['MAX','MIN','MEAN','CV'] 
    
    dfm = pd.DataFrame([['dset'+str(i+1),np.nan,np.nan,np.nan],[dates[i]['s'],np.nan,np.nan,np.nan], [dates[i]['e'],np.nan,np.nan,np.nan]])
    dfm.index = ['dataset','start','end']
    dfm.columns = ['MAX','MIN','MEAN','CV'] 
    
    dfr = pd.concat([dfm, dfc],axis=0)
    
    dfrs_dis.append(dfr)


dfda = dfrs_da[0]
for d in dfrs_da[1:]:
    dfda = pd.concat([dfda,d],axis=1)
    
dftemp = dfda['MEAN']
dftemp['MEANS ABSCV (%)'] =  np.round(np.abs(dftemp.std(axis=1)/dftemp.mean(axis=1))*100,2)
dfda['MEANS ABSCV (%)'] = dftemp['MEANS ABSCV (%)']


dfmae = dfrs_mae[0]
for d in dfrs_mae[1:]:
    dfmae = pd.concat([dfmae,d],axis=1)
    
dftemp = dfmae['MEAN']
dftemp['MEANS ABSCV (%)'] =  np.round(np.abs(dftemp.std(axis=1)/dftemp.mean(axis=1))*100,2)
dfmae['MEANS ABSCV (%)'] = dftemp['MEANS ABSCV (%)']

    
dfdis = dfrs_dis[0]
for d in dfrs_dis[1:]:
    dfdis = pd.concat([dfdis,d],axis=1)

dftemp = dfdis['MEAN']
dftemp['MEANS ABSCV (%)'] =  np.round(np.abs(dftemp.std(axis=1)/dftemp.mean(axis=1))*100,2)
dfdis['MEANS ABSCV (%)'] = dftemp['MEANS ABSCV (%)']

#%% SAVE RESULTS

path = ofolder+'datasets_compare.xlsx'

with pd.ExcelWriter(path) as writer:  
    dfda.to_excel(writer, sheet_name='MAX_DA')
    dfmae.to_excel(writer, sheet_name='MIN_MAE')
    dfdis.to_excel(writer, sheet_name='MIN_DIS')
