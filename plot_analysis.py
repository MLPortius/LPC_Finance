# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 17:19:58 2023

@author: Andriu
"""

#%% IMPORT LIBRARIES

import pandas as pd
import compress_pickle as cpickle
import plotly.express as px

from plotly.offline import plot

#%% SCRIPT SETUP

ds = 1

infolder = 'output/results/'
afolder = 'output/analysis/'
pfolder = 'output/plots/'

dset = 'set'+str(ds)
dset_path = dset+'/'

s1date = '2010-09-08'
s2date = '2016-08-23'
edate = '2022-08-08'

#%% LOAD DATA

with open(afolder+dset_path+'dfs.lzma','rb') as file:
    data = cpickle.load(file)
del(file)

da = data['da']
mae = data['mae']
dis = data['dis']

#%%

fig = px.s