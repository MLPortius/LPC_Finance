# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 10:26:31 2023

@author: Andriu
"""

#%% IMPORT LIBRARIES

import os
import pandas as pd
import numpy as np
import compress_pickle as cpickle
import argparse


#%% SCRIPT SETUP

parser = argparse.ArgumentParser(prog= 'LPC_Determinant_Analysis_Data_Preparation',
                                 description = 'Recopiles LPC data and creates dataframes',
                                 epilog = 'Created by Andriu')


parser.add_argument('-d','--dset', required=True, type=int, choices=[1,2,3])
args = parser.parse_args()

# ARGUMENTS

ds = args.dset

infolder = 'output/analysis/'
outfolder = 'output/determinants/'

dset = 'set'+str(ds)
dpath = dset+'/'


#%% FUNCTION DEFINITION

def ORDER_MDF(data, metric):

    mdf = data[metric]
    mdf.set_index('stock', drop=True, inplace=True)
    mdf = mdf.loc[:,[metric,'w_size','p_lags','vcv']]
    mdf.columns = [metric, 'WINDOW_SIZE', 'P_LAGS', 'VOLUME_COEF_VAR']
    
    return mdf

#%% DATA LOAD

print('\nLoading data...')

""" 
  
METRIC = b0 + b1 * VOLUMEN + b2 * TENDENCIA + b3i * INDUSTRIAi + b4 * PRECIO 
            + b5 * STD PERIODO + b6 * VOLATILIDAD HISTORICA + b7 * DOLLAR VOLUME OF SY CP 
            + b8 * DOLLAR VOLUME OF SY LP + b9 * BID-ASK SY + b10 * N. ANALISTAS SY
            + b11 * DISPERSION ANALISTAS + b12 * C.O.R SY PRICE + b13 * SY MARKET CAP 
            + b14 * RF LP + b15 * RF CP + b16 * SHORT SALE RATIO YY + b17 * SHORT SALE RATIO MM 
            + b18 * P_LAGS + b19 * W_SIZE

"""

mae = pd.read_excel(infolder + dpath + 'mae_opt.xlsx')
da = pd.read_excel(infolder + dpath + 'da_opt.xlsx')
dis = pd.read_excel(infolder + dpath + 'dis_opt.xlsx')

metric_dict = {'MAE':mae, 'DA':da, 'DIS':dis}

tickers = list(mae['stock'])


print('     ... grid info')

mae = ORDER_MDF(data=metric_dict, metric='MAE')
da = ORDER_MDF(data=metric_dict, metric='DA')
dis = ORDER_MDF(data=metric_dict, metric='DIS')


print('     ... histogram info')

histos = pd.read_excel(infolder + dpath + 'histos.xlsx')
cols = list(histos.columns)
cols[0] = 'stock'
cols[1] = 'STREAK_PROBA'
histos.columns = cols
histos.set_index('stock',drop=True, inplace=True)

mae = pd.concat([mae,histos['STREAK_PROBA']], axis=1)
da = pd.concat([da,histos['STREAK_PROBA']], axis=1)
dis = pd.concat([dis,histos['STREAK_PROBA']], axis=1)


print('     ... stock data')

lpc_datasets = cpickle.load('input/cpickle/lpc_datasets.lzma')
    
if ds == 1:
    
    lpc_datasets = lpc_datasets['set1']

    df_close = lpc_datasets[0]['close']
    df_vols = lpc_datasets[0]['vols']
    
    for d in lpc_datasets[1:]:
       df_close = pd.concat([df_close, d['close']], axis=1) 
       df_vols = pd.concat([df_vols, d['vols']], axis=1)
       
    data = {'close':df_close, 'vols':df_vols}
        
        
elif ds == 2:
    
    lpc_datasets = lpc_datasets['set2']

    df_close = lpc_datasets[0]['close']
    df_vols = lpc_datasets[0]['vols']
    
    for d in lpc_datasets[1:]:
       df_close = pd.concat([df_close, d['close']], axis=1) 
       df_vols = pd.concat([df_vols, d['vols']], axis=1)

    data = {'close':df_close, 'vols':df_vols}
    
    
elif ds == 3:
    
    data1 = lpc_datasets['set1']
    
    df_close_1 = data1[0]['close']
    df_vols_1 = data1[0]['vols']
    
    for d in data1[1:]:
       df_close_1 = pd.concat([df_close_1, d['close']], axis=1) 
       df_vols_1 = pd.concat([df_vols_1, d['vols']], axis=1)
    
    data2 = lpc_datasets['set2']
    
    df_close_2 = data2[0]['close']
    df_vols_2 = data2[0]['vols']
    
    for d in data2[1:]:
       df_close_2 = pd.concat([df_close_2, d['close']], axis=1) 
       df_vols_2 = pd.concat([df_vols_2, d['vols']], axis=1)
    
    df_close = pd.concat([df_close_1, df_close_2], axis=0)
    df_vols = pd.concat([df_vols_1, df_vols_2], axis=0)
    
    data = {'close':df_close, 'vols':df_vols}


mean_prices = df_close.mean(axis=0).to_frame()
desv_prices = df_close.std(axis=0).to_frame()

mean_prices.columns = ['MEAN_PRICE']
desv_prices.columns = ['PRICE_STD']

mae = pd.concat([mae, mean_prices, desv_prices], axis=1)
da = pd.concat([da, mean_prices, desv_prices], axis=1)
dis = pd.concat([dis, mean_prices, desv_prices], axis=1)

#%% EXPORT DETERMINANT DATA

print('\nExporting data...')

det_dict = {'mae':mae, 'da':da, 'dis':dis}

with open(outfolder + dpath + 'det_data_dict.lzma','wb') as file:
    cpickle.dump(det_dict, file, compression='lzma')

mae.to_excel(outfolder + dpath + 'mae/' + 'mae_data.xlsx')
da.to_excel(outfolder + dpath + 'da/' + 'da_data.xlsx')
dis.to_excel(outfolder + dpath + 'dis/' + 'dis_data.xlsx')

print('     ...Done!')
