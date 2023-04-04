# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 11:24:57 2023

@author: Andriu
"""

#%% IMPORT LIBRARIES

import os
import numpy as np
import pandas as pd
import compress_pickle as cpickle

import time

import warnings
warnings.filterwarnings('ignore')

#%% SCRIPT SETUP

# import argparse

# parser = argparse.ArgumentParser(prog= 'LPC_tickers_PCR',
#                                  description = 'Scrap Options Market dataset and gets Put Call Ratio',
#                                  epilog = 'Created by Andriu')

# parser.add_argument('-d', kwargs)
# parser.add_argument('-d','--dset', required=True, type=int, choices=[1,2,3])
# parser.add_argument('-m','--metrics', required=True, type=str, help='use da-mae-dis')
# parser.add_argument('-g','--grid',required=True, type=bool)

# args = parser.parse_args()

# # ARGUMENTS
# ds = args.dset
# met = args.metrics
# grid_flag = args.grid


#%% SETUP

# MINIMUN RAM 10 GB

infolder1 = 'input/info/'
infolder2 = 'input/pcr/'
outfolder = 'input/pcr/tickers/'
outfolder2 = 'input/cpickle/'

#%% READ TICKERS

print('\nReading LPC tickers ...')

with open(infolder1+'lpc_tickers.txt','r') as file:
    tickers = file.read()
    tickers = tickers.split(';')
    tickers = tickers[:-1]

del(file)

print('     ...done!')


#%% CHECK CSV

print('\nChecking files ...')

if 'open_interest.zip' in os.listdir(infolder2):
    error = False
else:
    error = True


done_list = [x.split('.')[0] for x in os.listdir(outfolder)]

if len(done_list) == len(tickers):
    done = True

else:
    for t in done_list:
        tickers.pop(tickers.index(t))

print('     ...done!')


#%% FUNCITON DEFINITION

def LOAD_DATA(size):
    
    iter_csv = pd.read_csv(infolder2+'open_interest.csv', iterator=True, chunksize=1500000, parse_dates=['date'])  
    data = pd.concat([chunk for chunk in iter_csv])
    
    return data

def READ_PCR(tag, dataset):
    
    # READ DATA
    # iter_csv = pd.read_csv(infolder2+'open_interest.csv', iterator=True, chunksize=1500000, parse_dates=['date'])
    # data = pd.concat([chunk[chunk['ticker'] == tag] for chunk in iter_csv])

    df = dataset[dataset['ticker']==tag]
    
    # PUT CALL RATIO
    puts = df[df['cp_flag']=='P']
    calls = df[df['cp_flag']=='C']

    puts = puts.loc[:,['date','volume','open_interest']]
    calls = calls.loc[:,['date','volume','open_interest']]

    puts.columns = ['date']+['P_'+x for x in list(puts.columns)[1:]]
    calls.columns = ['date']+['C_'+x for x in list(calls.columns)[1:]]
    
    pcr = pd.merge(puts, calls, right_on='date', left_on='date',how='inner')
    pcr.sort_values('date', ascending=True, inplace=True)
    
    # pcr.set_index('date', drop=True, inplace=True)
    # pcr.sort_index(axis=0, ascending=True, inplace=True)
    
    pcr['PC_RATIO'] = pcr['P_volume']/pcr['C_volume']

    return pcr

def RETRIEVE_PCR(tag):    
    
    df = pd.read_excel(outfolder+tag+'.xlsx', index_col='date')
    
    return df
    
    
#%% PROCESS DATA

if error:
    
    print('\nERROR: Manually unzip open_interest.zip first...')

else:
    
    if done:
        
        print('\nRetrieving excel files ...')
        
        done_list = [x.split('.')[0] for x in os.listdir(outfolder)]

        pcrs = {}
        
        n = len(done_list)
        
        for i in range(n):
            
            t = done_list[i]
            p = np.round(((i+1)/n)*100,2)
            
            print('\n',t,' - ',p,'% ...')
        
            pcrs[t] = RETRIEVE_PCR(t)
        
        with open(outfolder2+'putcallratio.lzma','wb') as file:
            cpickle.dump(pcrs, file, compression='lzma')
        
        print('     ...done!')
        
    else:
        
        print('\nLoading dataset ...')
        
        start = time.time()
        
        data = LOAD_DATA(1000000)
        
        end = time.time()
        
        seconds = end - start
        
        minutes = np.round(seconds/60, 2)
    
        print('     ... done in',minutes,'minutes!')
    
    
        print('\nProcessing data and saving files...')
        
        n = len(tickers)
        
        for i in range(n):
            
            start = time.time()
            
            t = tickers[i]        
            p = np.round(((i+1)/n) * 100, 2)
            
            print('\n',t,' - ', p, '% ...')
            
            res = READ_PCR(t, data)
            
            res.to_excel(outfolder+t+'.xlsx', index=False)
            
            # pcrs[t] = READ_PCR(t)
            
            # with open(outfolder+'putcallratio.lzma','wb') as file:
            #     cpickle.dump(pcrs, file, compression='lzma')
            
            end = time.time()
            
            seconds = end - start
        
            minutes = np.round(seconds/60, 2)
            
            print('     ...',minutes,'minutes!')
            
        print('\n     ...done!')
        
        
        print('\nRetrieving excel files ...')
        
        done_list = [x.split('.')[0] for x in os.listdir(outfolder)]

        pcrs = {}
        
        n = len(done_list)
        
        for i in range(n):
            
            t = done_list[i]
            p = np.round(((i+1)/n)*100,2)
            
            print('\n',t,' - ',p,'% ...')
        
            pcrs[t] = RETRIEVE_PCR(t)
        
        with open(outfolder2+'putcallratio.lzma','wb') as file:
            cpickle.dump(pcrs, file, compression='lzma')
        
        print('\n     ...done!')
