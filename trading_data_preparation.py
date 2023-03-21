# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 15:10:31 2023

@author: Andriu
"""

#%% IMPORT LIBRARIES

import os
import pandas as pd
import numpy as np
import compress_pickle as cpickle
import argparse


#%% SCRIPT SETUP

parser = argparse.ArgumentParser(prog= 'LPC_Trading_Data_Preparation',
                                 description = 'Recopiles and Saves the necesary data for Trading Simulation',
                                 epilog = 'Created by Andriu')


parser.add_argument('-d','--dset', required=True, type=int, choices=[1,2,3])
parser.add_argument('-m','--metrics', required=True, type=str, help='use da-mae-dis')
parser.add_argument('-g','--grid',required=True, type=bool)

args = parser.parse_args()

# ARGUMENTS
ds = args.dset
met = args.metrics
grid_flag = args.grid

mets = met.split('-')

da_flag = False
mae_flag = False
dis_flag = False

if 'da' in mets:
    da_flag = True
    
if 'mae' in mets:
    mae_flag = True

if 'dis' in mets:
    dis_flag = True

dset = 'set'+str(ds)
dset_path = dset+'/'


infolder1 = 'input/cpickle/'
infolder2 = 'output/analysis/' + dset_path

outfolder = 'output/analysis/' + dset_path
outfolder2 = 'output/trading_simulation/' + dset_path 

#%% GET PRICES

print('\nLoading data...')

prices = cpickle.load(infolder1 + 'tickers_data.lzma')

mae = pd.read_excel(infolder2 +  'mae_opt.xlsx')
mae.set_index('stock',drop=True,inplace=True)

da = pd.read_excel(infolder2 + 'da_opt.xlsx')
da.set_index('stock',drop=True,inplace=True)

dis = pd.read_excel(infolder2 + 'dis_opt.xlsx')
dis.set_index('stock',drop=True,inplace=True)

tickers = list(mae.index)

#%% GET OPT CONFIGS

mae_configs = mae.loc[:, ['w_size', 'p_lags']]
da_configs = da.loc[:, ['w_size', 'p_lags']]
dis_configs = dis.loc[:, ['w_size','p_lags']]

mae_configs_dict = {}
da_configs_dict = {}
dis_configs_dict = {}

print('\nGetting opt configs for lpc rolling...')
for t in tickers:
    mae_configs_dict[t] = list(mae_configs.loc[t,:])
    da_configs_dict[t] = list(da_configs.loc[t,:])
    dis_configs_dict[t] = list(dis_configs.loc[t,:])

print('     ...exporting opt configs')

opt_configs = {'mae':mae_configs_dict, 
               'da':da_configs_dict, 
               'dis':dis_configs_dict}

with open(outfolder + 'opt_configs.lzma', 'wb') as file:
    cpickle.dump(opt_configs, file, compression='lzma')
    

#%% PRICE FILTER


# NECESITO: CLOSE, HIGH, LOW, OPEN, LAST, PREDS

print('\nRecovering prices ...')

prices_dict = {}

for t in tickers:
    
    df_price = prices[t]
    df_price = df_price.iloc[:, :4]

    if ds == 3:
        df_price = df_price
    elif ds == 2:
        df_price = df_price.iloc[1500:,:]
    elif ds == 1:
        df_price = df_price.iloc[:1500,:]

    prices_dict[t] = df_price
    
del(df_price, t)

print('     ... Done!')


print('\nRecovering last preds ...')

mae_lasts_dict = {}
da_lasts_dict = {}
dis_lasts_dict = {}

for t in tickers:
    
    v1 = mae.loc[t,'last_pred']
    v2 = da.loc[t, 'last_pred']
    v3 = dis.loc[t, 'last_pred']
    
    mae_lasts_dict[t] = v1
    da_lasts_dict[t] = v2
    dis_lasts_dict[t] = v3

del(t, v1, v2, v3)

print('     ... Done!')

#%% MINI GRID

if grid_flag == True:
    # ------------------------------------------------------------------
    
    print('\nRetrieving lpc datasets...')
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
    
    print('     ... Done!')
    
    
    # ------------------------------------------------------------------
    
    print('\nRunning mini grid...')
    
    import time
    from classes import grid
    LPCgrid = grid.CLASS
    
    if mae_flag == True:
        
        mae_results_dict = {}
        
        timelapse = []
        n = len(tickers)
        
        for i in range(n):
            
            start = time.time()
            
            percent = np.round(((i+1)/len(tickers)) * 100,2)
            t = tickers[i]
            
            if len(timelapse) == 0: 
                print ('\n    ...', percent, '% - ??? minutes to finish -',t)
            
            else:
                estimated = np.round((np.mean(timelapse)*(n-i)) / 60, 4)
                print('\n     ...', percent, '% -',estimated,'minutes to finish -',t)
        
            mae_wg = mae_configs_dict[t][0]
            mae_pg = mae_configs_dict[t][1]
            
            mae_lpc = LPCgrid(data, t, [mae_wg], [mae_pg])
            mae_lpc.grid_search()
            mae_lpc.global_summary()
            mae_res = mae_lpc.roll_list[0].results
            
            mae_results_dict[t] = mae_res
        
            end = time.time()
            elapsed = end - start
            timelapse.append(elapsed)
        
        
        with open(outfolder2 + 'mae_grid_results.lzma' , 'wb') as file:
            cpickle.dump(mae_results_dict, file, compression='lzma')
    
    
    if da_flag == True:
        
        da_results_dict = {}
        
        timelapse = []
        n = len(tickers)
        
        for i in range(n):
            
            start = time.time()
            
            percent = np.round(((i+1)/len(tickers)) * 100,2)
            t = tickers[i]
            
            if len(timelapse) == 0: 
                print ('\n    ...', percent, '% - ??? minutes to finish -',t)
            
            else:
                estimated = np.round((np.mean(timelapse)*(n-i)) / 60, 4)
                print('\n     ...', percent, '% -',estimated,'minutes to finish -',t)
        
            da_wg = da_configs_dict[t][0]
            da_pg = da_configs_dict[t][1]
            
            da_lpc = LPCgrid(data, t, [da_wg], [da_pg])
            da_lpc.grid_search()
            da_lpc.global_summary()
            da_res = da_lpc.roll_list[0].results
            
            da_results_dict[t] = da_res
        
            end = time.time()
            elapsed = end - start
            timelapse.append(elapsed)
        
        with open(outfolder2 + 'da_grid_results.lzma' , 'wb') as file:
            cpickle.dump(da_results_dict, file, compression='lzma')
        
    
    if dis_flag == True:
        
        dis_results_dict = {}
        
        timelapse = []
        n = len(tickers)
        
        for i in range(n):
            
            start = time.time()
            
            percent = np.round(((i+1)/len(tickers)) * 100,2)
            t = tickers[i]
            
            if len(timelapse) == 0: 
                print ('\n    ...', percent, '% - ??? minutes to finish -',t)
            
            else:
                estimated = np.round((np.mean(timelapse)*(n-i)) / 60, 4)
                print('\n     ...', percent, '% -',estimated,'minutes to finish -',t)
        
            dis_wg = dis_configs_dict[t][0]
            dis_pg = dis_configs_dict[t][1]
        
            dis_lpc = LPCgrid(data, t, [dis_wg], [dis_pg])
            dis_lpc.grid_search()
            dis_lpc.global_summary()
            dis_res = dis_lpc.roll_list[0].results
        
            dis_results_dict[t] = dis_res
        
            end = time.time()
            elapsed = end - start
            timelapse.append(elapsed)
        
        with open(outfolder2 + 'dis_grid_results.lzma' , 'wb') as file:
            cpickle.dump(dis_results_dict, file, compression='lzma')

#%% EXPORT TRADING DATA 

print('\nRecovering y predictions ...')

mae_preds = cpickle.load('output/trading_simulation/'+dset_path+'mae_grid_results.lzma')
da_preds = cpickle.load('output/trading_simulation/'+dset_path+'da_grid_results.lzma')
dis_preds = cpickle.load('output/trading_simulation/'+dset_path+'dis_grid_results.lzma')

print('     ...Done!')

print('\nBuilding trading data dictos...')

mae_opt_dict = {}
da_opt_dict = {}
dis_opt_dict = {}

for t in tickers:
    
    mae_df = mae_preds[t]
    mae_df['PREDICTION'] = mae_df['NORM_PREDICTION']*mae_df['STD'] + mae_df['MEANS']
    mae_stock_dicto = {'prices':prices_dict[t].loc[mae_df.index,:], 'last':mae_lasts_dict[t], 'ypred':mae_df['PREDICTION'], 'config':mae_configs_dict[t]}
    mae_opt_dict[t] = mae_stock_dicto
    
    da_df = da_preds[t]
    da_df['PREDICTION'] = da_df['NORM_PREDICTION']*da_df['STD'] + da_df['MEANS']
    da_stock_dicto = {'prices':prices_dict[t].loc[da_df.index,:], 'last':da_lasts_dict[t], 'ypred':da_df['PREDICTION'], 'config':da_configs_dict[t]}
    da_opt_dict[t] = da_stock_dicto
    
    dis_df = dis_preds[t]
    dis_df['PREDICTION'] = dis_df['NORM_PREDICTION']*dis_df['STD'] + dis_df['MEANS']
    dis_stock_dicto = {'prices':prices_dict[t].loc[dis_df.index,:], 'last':dis_lasts_dict[t], 'ypred':dis_df['PREDICTION'], 'config':dis_configs_dict[t]}
    dis_opt_dict[t] = dis_stock_dicto

print('     ...Done!')


print('\nSaving trading data dictos ...')

with open(outfolder2 + 'mae_trading_data.lzma' , 'wb') as file:
    cpickle.dump(mae_opt_dict, file, compression='lzma')

with open(outfolder2 + 'dis_trading_data.lzma' , 'wb') as file:
    cpickle.dump(dis_opt_dict, file, compression='lzma')

with open(outfolder2 + 'da_trading_data.lzma' , 'wb') as file:
    cpickle.dump(da_opt_dict, file, compression='lzma')

print('     ...Done!')