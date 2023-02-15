# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 05:50:43 2023

@author: Andriu
"""

#%% IMPORTAR LIBRERIAS

import pandas as pd
import compress_pickle as cpickle
import argparse

#%% SCRIPT SETUP

parser = argparse.ArgumentParser(prog= 'LPC_Results_Analysis',
                                 description = 'Join results and optimize metrics',
                                 epilog = 'Created by Andriu')

parser.add_argument('-d','--dset', required=True, type=int, choices=[1,2,3])
args = parser.parse_args()

# ARGUMENTS
ds = args.dset


infolder = 'output/results/'
afolder = 'output/analysis/'

dset = 'set'+str(ds)
dset_path = dset+'/'

#%% LOAD DATA

with open(infolder+dset+'.lzma','rb') as file:
    data = cpickle.load(file)
del(file)

tickers = list(data.keys())
tickers.sort()

grids = {}
for t in tickers:
    grids[t] = data[t]['grid']
del(t)

vols = {}
for t in tickers:
    vols[t] = data[t]['vm']
del(t)

hist = {}
for t in tickers:
    hist[t] = data[t]['hm']
del(t)

#%% FUNCTIONS 

def sort_grid(ind, gdict, metric, asc=False):

    df = gdict[ind]
    df.sort_values(metric, ascending=asc, inplace=True)
    d = df.iloc[0,:].to_frame().T
    
    return d

#%% OPTIMIZE DA

print('\nOptimizing DA...')
da_df = sort_grid(ind=tickers[0], gdict=grids, asc=False, metric='DA')

for t in tickers[1:]:
    d = sort_grid(ind=t, gdict=grids, asc=False, metric='DA')
    da_df = pd.concat([da_df,d], axis=0)

del(d, t)

da_df.set_index('stock',inplace=True, drop=True)

#%% OPTIMIZE MAE

print('\nOptimizing MAE...')
mae_df = sort_grid(ind=tickers[0], gdict=grids, asc=True, metric='MAE')

for t in tickers[1:]:
    d = sort_grid(ind=t, gdict=grids, asc=True, metric='MAE')
    mae_df = pd.concat([mae_df,d], axis=0)

del(d, t)

mae_df.set_index('stock',inplace=True, drop=True)

#%% OPTIMIZE DIS

print('\nOptimizin DIS...')

dis_df = sort_grid(ind=tickers[0], gdict=grids, asc=True, metric='DIS')

for t in tickers[1:]:
    d = sort_grid(ind=t, gdict=grids, asc=True, metric='DIS')
    dis_df = pd.concat([dis_df,d], axis=0)

del(d, t)

dis_df.set_index('stock',inplace=True, drop=True)

#%% ADD VOLS

print('\nAdding VOLUME metrics...')

v_df = vols[tickers[0]]
for t in tickers[1:]:
    v_df = pd.concat([v_df,vols[t]], axis=0)
del(t)

v_df.set_index('stock', inplace=True)

da_df = pd.concat([da_df, v_df], axis=1)
mae_df = pd.concat([mae_df, v_df], axis=1)
dis_df = pd.concat([dis_df, v_df], axis=1)

#%% ADD HISTOS

print('\nAdding HISTOGRAMS metrics...')

h_df = hist[tickers[0]]
for t in tickers[1:]:
    h_df = pd.concat([h_df,hist[t]], axis=0)
del(t)

h_df.set_index('stock', inplace=True)

da_df = pd.concat([da_df, h_df], axis=1)
mae_df = pd.concat([mae_df, h_df], axis=1)
dis_df = pd.concat([dis_df, h_df], axis=1)

da_df = da_df.infer_objects()
mae_df = mae_df.infer_objects()
dis_df = dis_df.infer_objects()

#%% SAVE RESULTS

print('\nSaving Analysis Results...')

dicto = {'da':da_df,'mae':mae_df, 'dis':dis_df}

da_df.to_excel(afolder+dset_path+'da_opt.xlsx')
mae_df.to_excel(afolder+dset_path+'mae_opt.xlsx')
dis_df.to_excel(afolder+dset_path+'dis_opt.xlsx')

with open(afolder+dset_path+'dfs.lzma','wb') as file:
    cpickle.dump(dicto, file, compression='lzma')

print('     ...DONE!')
