# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 20:49:55 2023

@author: Andriu
"""

#%% IMPORT LIBRARIES

import eikon
import compress_pickle as cpickle
import pandas as pd
import time


#%% SETUP SCRIPT

infolder = 'input/info/'
outfolder = 'input/eikon/' 

# Eikon
with open(outfolder+'eikon.api','r') as file:
    api_id = file.read()
    
eikon.set_app_key(api_id)


#%% LOAD DATA

with open(infolder+'lpc_tickers.txt','r') as file:
    lpc = file.read()
    lpc = lpc.split(';')

with open(infolder+'ric_tickers.txt','r') as file:
    ric = file.read()
    ric = ric.split(';')
    ric = ric[:-1]
    ric = sorted(ric)

ric_lpc = []
for r in ric:
    for t in lpc:
        print(t,'in',r,'...')
        if t in r:
            ric_lpc.append(r)
            print('...ok')

ric[0] in lpc[0] 