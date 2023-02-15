# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 05:49:17 2023

@author: Andriu
"""

#%% IMPORT LIBRARIES

import os
import pandas as pd
import numpy as np
import compress_pickle as cpickle
import argparse


#%% SCRIPT SETUP

parser = argparse.ArgumentParser(prog= 'LPC_Retrive_Results',
                                 description = 'Recopiles and Saves the results from a Rolling LPC Model Grid',
                                 epilog = 'Created by Andriu')


parser.add_argument('-d','--dset', required=True, type=int, choices=[1,2,3])
args = parser.parse_args()

# ARGUMENTS
ds = args.dset


infolder = 'output/results/'
outfolder = 'output/results/'

dset = 'set'+str(ds)
dset_path = dset+'/'

#%% RETRIEVE RESULTS

slices = os.listdir(infolder+dset_path)

r_files = []
r_ind = []

for s in slices:
    
    print(s,'...')
    
    path = infolder+dset_path+s+'/'
    
    r = os.listdir(path)
    r.pop(r.index('.gitkeep'))

    ri = [x.split('.')[0] for x in r]
    rf = [path+x for x in r]
    
    r_ind = r_ind + ri
    r_files = r_files + rf
    
    print(' ...Done!')

del(s, r, path, ri, rf)


files = {}
for i in range(len(r_files)):
    print(np.round(i/len(r_files)*100,2),'% ...')
    dicto = {'grid':pd.read_excel(r_files[i], sheet_name='GRID'),
             'vm':pd.read_excel(r_files[i], sheet_name='VMETRICS'),
             'hm':pd.read_excel(r_files[i], sheet_name='HMETRICS'),
             'ht':pd.read_excel(r_files[i], sheet_name='TOTAL_HS'),
             'hp':pd.read_excel(r_files[i], sheet_name='POSITIVE_HS'),
             'hn':pd.read_excel(r_files[i], sheet_name='NEGATIVE_HS')}
    
    files[r_ind[i]] = dicto

del(i, dicto)


with open(infolder+dset+'.lzma','wb') as file:
    cpickle.dump(files, file, compression='lzma')
