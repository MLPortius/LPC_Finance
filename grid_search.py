# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 20:37:56 2023

@author: Andriu
"""

#%% IMPORTAR LIBRERIAS

import os
import time
import pandas as pd
import numpy as np
import compress_pickle as cpickle

from classes import grid
LPCgrid = grid.CLASS

#%% SCRIPT SETUP

import argparse

parser = argparse.ArgumentParser(prog= 'LPC_grid_search',
                                 description = 'Analyse a Time Serie with Rolling LPC model',
                                 epilog = 'Created by Andriu')

parser.add_argument('-s','--slice', required=True, type=int, choices=[1,2,3,4,5,6,7,8,9,10])
parser.add_argument('-d','--dset', required=True, type=int, choices=[1,2,3])
parser.add_argument('-g','--grid', required=True, type=str, choices=['short','full'])
parser.add_argument('--git', required=True, type=int, choices=[0,1])

args = parser.parse_args()

# ARGUMENTS
ds = args.dset
sl = args.slice
g = args.grid
git = args.git

#%% CONFIGURAR ENTORNO

infolder = 'input/cpickle/'
outfolder = 'output/results/'

dset = 'set'+str(ds)
slce = 'slice'+str(sl)

slice_path = 'slice'+str(sl)+'/'
dset_path = dset+'/'

#%% CARGAR DATOS

with open(infolder+'slice_tickers'+'.lzma','rb') as file:
    slice_tickers = cpickle.load(file)
    
with open(infolder+'lpc_datasets'+'.lzma','rb') as file:
    lpc_datasets = cpickle.load(file)

tickers = slice_tickers[sl-1]

if ds == 3:    
    data1 = lpc_datasets['set1'][sl-1]
    data2 = lpc_datasets['set2'][sl-1]
    data = {'close':pd.concat([data1['close'], data2['close']],axis=0),
            'vols':pd.concat([data1['vols'], data2['vols']],axis=0)}
else:    
    data = lpc_datasets[dset][sl-1]


if g == 'short':
    wg = [100,250,500]
    pg = [5, 25, 50]
    
elif g == 'full':
    wg = [30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 200, 250, 300, 400, 500, 600]
    pg = [1,2,3,4,5,6,7,8,9,10,12,15,20,25,30]

    
#%% INICIALIZAR GRILLA

ready = os.listdir(outfolder+dset_path+slice_path)
ready.pop(ready.index('.gitkeep'))
ready = [x.split('.')[0] for x in ready]

for t in tickers:
    
    if not t in ready:    
        start = time.time()
        
        print('\nSTARTING',t,'GRID SEARCH...')
        
        lpc = LPCgrid(data, t, wg, pg)
        lpc.grid_search()
        lpc.global_summary()
        lpc.add_volumes()
        lpc.add_histograms()
        
        gsum = lpc.gsum
        hdf = lpc.hdf
        vdf = lpc.vdf
        histograms = lpc.hs
        
        ht = histograms['t']
        hp = histograms['p']
        hn = histograms['n']
        
        path = outfolder+dset_path+slice_path+t+'.xlsx'
        
        with pd.ExcelWriter(path) as writer:  
            gsum.to_excel(writer, sheet_name='GRID')
            vdf.to_excel(writer, sheet_name='VMETRICS')
            hdf.to_excel(writer, sheet_name='HMETRICS')
            ht.to_excel(writer, sheet_name='TOTAL_HS')
            hp.to_excel(writer, sheet_name='POSITIVE_HS')
            hn.to_excel(writer, sheet_name='NEGATIVE_HS')
        
        if git == 1:
            
            git_branch = dset+'/'+slce
            
            os.system('git add .')
            os.system('git commit -m'+' '+'"grid result - '+dset+' - '+slce+' - '+t+'"')
            os.system("git"+" "+"-u push lpc"+" "+git_branch)
            
        end = time.time()
        elapsed = np.round((end-start)/3600,2)
        
        print('\nGRID TIME: ',elapsed,'HRS')
        print('     ...DONE!')
    
        
    else:
        print('\n')
        print(t, 'Already Done...')