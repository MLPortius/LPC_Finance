# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 03:39:57 2023

@author: Andriu
"""

#%% IMPORT LIBRARIES

import os
import numpy as np
import pandas as pd
import compress_pickle as cpickle

import warnings 
warnings.filterwarnings('ignore')

import argparse


#%% SCRIPT SETUP

parser = argparse.ArgumentParser(prog= 'LPC_HURST_EXPONENT_METRICS',
                                  description = 'Recopiles LPC data and calculates hurst exponent',
                                  epilog = 'Created by Andriu')

parser.add_argument('-d','--dset', required=True, type=int, choices=[1,2,3])
args = parser.parse_args()

# ARGUMENTS

ds = args.dset


#%% ENVIRONMENT SETUP

# ds = 1

infolder = 'input/info/'
outfolder = 'input/hurst/'

dset = 'set'+str(ds)
dpath = dset+'/' 


#%% FUNCTION DEFINITION

def GET_HURST(serie, max_lag=21):
        
    lags = list(range(2, max_lag))
    
    tau = []
    for l in lags:
        _ = serie - serie.shift(l)
        _.dropna(axis=0, inplace=True)
        tau.append(_.std())
        
    tau = np.asarray(tau)
    lags = np.asarray(lags)
    
    reg = np.polyfit(np.log(lags), np.log(tau), 1)

    return reg[0]


def GET_HURST_GRIDMEAN(time_serie):
    
    # 1TM, 2TM, 3TM, 5TM, 7TM, 9TM, 1TY, 1.2TY, 1.5TY, 1.7TY, 2TY 
    grid = [21, 42, 63, 147, 189, 252, 315, 378, 441, 504]
    
    hs = []
    for mlag in grid:
        hs.append(GET_HURST(time_serie, max_lag = mlag))
        
    hs = pd.Series(hs)
    hsm = hs.mean()
    
    return hsm


def GET_HURST_CLASS(x):
    
    x2 = np.round(x, 1)
    
    if x2 == 0.5:
        y = 'RANDWALK'
        
    elif x2 < 0.5:
        y = 'MEANREV'
        
    elif x2 > 0.5:
        y = 'TRENDING'
        
    else:
        y = 'ERROR'
    
    return y 


def GET_PRICES_DICT(i, save=False):
    
    dicto = cpickle.load('input/cpickle/lpc_datasets.lzma')

    d1 = dicto['set1']
    d2 = dicto['set2']
    
    d1 = [x['close'] for x in d1]
    d2 = [x['close'] for x in d2]
    
    df1 = d1[0]
    for d in d1[1:]:
        df1 = pd.concat([df1, d], axis=1)
    
    df2 = d2[0]
    for d in d2[1:]:
        df2 = pd.concat([df2, d], axis=1)
    
    df3 = pd.concat([df1, df2], axis=0)
    
    if save:
        dts = {'set1':df1.index, 'set2':df2.index, 'set3':df3.index}
        with open('input/info/datasets_dates.lzma', 'wb') as file:
            cpickle.dump(dts, file, compression='lzma')
    
    else:
        
        if i == 1:
            output = {}
            for c in list(df1.columns):
                output[c] = df1.loc[:,c]
        
        elif i == 2:
            output = {}
            for c in list(df2.columns):
                output[c] = df2.loc[:,c]
        
        elif i == 3:
            output = {}
            for c in list(df3.columns):
                output[c] = df3.loc[:,c]
                
        else:
            output = 'error'
            
        return output
    

#%% LOAD DATA

print('\nLoading tickers...')

with open(infolder+'lpc_tickers.txt','r') as file:
    lpc = file.read()
    lpc = lpc.split(';')
    lpc = lpc[:-1]

print('     ...done!')


print('\nLoading prices...')

prices = GET_PRICES_DICT(ds)

print('     ...done!')


print('\nLoading done hursts...')


files = os.listdir(outfolder + dpath)
files.pop(files.index('.gitkeep'))

done_list = [x.split('.')[0] for x in files]

for dl in done_list:
    lpc.pop(lpc.index(dl))
    
print('     ...done!')
    

#%% HUSRT EXPONENT

for tag in lpc:

    print('\n',tag,'...')
    
    d = prices[tag]
    
    # METRIC 1 - GRIDMEAN
    hgm = GET_HURST_GRIDMEAN(d)
    
    # METRIC 2 - GRIDMEAN CLASS
    hgmc = GET_HURST_CLASS(hgm)
    
    # METRIC 3 - ROLLING MEAN
    r = d.rolling(252).apply(GET_HURST)
    r = r.to_frame()
    r.index = d.index
    r.columns = ['HURST']
    r = pd.Series(r['HURST'])
    r.dropna(inplace=True)
    
    hrmean = r.mean()
    hrlast = r[-1]
    
    # METRIC 4 - ROLLING CLASS PROBA
    r = r.to_frame()
    r['CAT'] = r['HURST'].apply(GET_HURST_CLASS)
    
    dum = pd.get_dummies(r['CAT'], prefix='HURST',dtype=int)
    
    probas = dum.mean()
    
    # METRIC 5 - ROLLING CLASS DAYS
    days = dum.sum()
    
    # METRIC 6 - DISTANCE FROM RANDOM WALK
    r['DFRW'] = np.abs(r['HURST'] - 0.5)
    
    dfrw = r['DFRW'].mean()
    
    dicto = {'hgm':hgm, 'hgmc':hgmc, 'rdf':r, 'rmean':hrmean,'rlast':hrlast, 'pr':probas, 'dy':days, 'dfrw':dfrw, 'dum':dum}
    
    df1 = pd.DataFrame([dicto['hgm'], dicto['hgmc'], dicto['rlast'], dicto['rmean'],dicto['dfrw']])
    df1 = df1.T
    df1.columns = ['HURST_GRIDMEAN','HURST_GRIDMEAN_CLASS', 'HURST_RLAST', 'HURST_RMEAN', 'HURST_DFRW']
    
    df2 = dicto['pr'].to_frame().T
    df2.columns = [x +'_PROBA' for x in df2]
    
    df3 = dicto['dy'].to_frame().T
    df3.columns = [x +'_DAYS' for x in df3]
    
    df4 = pd.concat([df1, df2, df3], axis=1)
    
    df5 = dicto['rdf']
    df5.reset_index(drop=False, inplace=True)
    
    df6 = dicto['dum']
    df6.reset_index(drop=False, inplace=True)
    
    
    with pd.ExcelWriter(outfolder + dpath + tag + '.xlsx') as writer:
        df4.to_excel(writer, sheet_name="METRICS", index=False)
        df5.to_excel(writer, sheet_name="SERIES", index=False)
        df6.to_excel(writer, sheet_name="DUMMIES", index=False)
        
    del(d, hgm, hgmc, r, hrmean, hrlast, dum, probas, days, dicto, df1, df2, df3, df4, df5, df6, writer)