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

parser = argparse.ArgumentParser(prog= 'LPC_Retrive_Histogram_Proba',
                                 description = 'Recopiles and Saves the proba of streak > 1',
                                 epilog = 'Created by Andriu')


parser.add_argument('-d','--dset', required=True, type=int, choices=[1,2,3])
args = parser.parse_args()

# ARGUMENTS
ds = args.dset


infolder = 'output/results/'
outfolder = 'output/analysis/'

dset = 'set'+str(ds)
dset_path = dset+'/'

def CAT(x):
    if x > 1:
        y = 1
    else:
        y = 0
    return y

#%% HISTOGRAMS PROBAS

files = []

for f in os.listdir(infolder):
    
    if '.lzma' in f:
        print(f)
        dicto = cpickle.load(infolder+f)
        files.append(dicto) 
        
files = files[ds-1]

hts = {}
hps = {}
hns = {}

for k in files.keys():
    print(k,'...')
    hts[k] = files[k]['ht']
    hps[k] = files[k]['hp']
    hns[k] = files[k]['hn']


probas = {}
for k in hts.keys():
    print(k, '...')
    df = hts[k]
    df = df.iloc[:,1:]
    df['cat'] = df['magnitud'].apply(CAT)
    df['proba2'] = df['cat']*df['proba']
    proba = df['proba2'].sum()
    maxi = df['magnitud'].max()
    dicto = {'pr':proba, 'max':maxi}
    probas[k] = dicto

del(df, dicto, maxi, proba, k)

df = pd.DataFrame(probas).T
df.columns = ['STREAK PROBA', 'MAX STREAK']

df.to_excel(outfolder+dset_path+'histos.xlsx')