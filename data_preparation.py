# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 23:47:16 2023

@author: Andriu
"""

import pandas as pd
import pickle
import compress_pickle as cpickle

infolder = 'input/'
outfolder = 'output/'

data = pd.read_excel(infolder+'sp500_market_data.xlsx')
slices_ind = list(data.columns)


#%% TICKERS
slices_tickers = {}
for s in slices_ind:
    print(s,'...')
    tickers = []
    for t in list(data[s].dropna(axis=0).values):
        tickers.append(t.split('.')[0])
    slices_tickers[s] = tickers
del(tickers, s, t)
    

#%% SHEETS DATA
slices_data = {}
for s in slices_ind:
    print(s,'...')
    sm = s.lower().replace(' ','_')
    d = pd.read_excel(infolder+'sp500_market_data.xlsx',sheet_name=sm)
    d.set_index('Date',drop=True,inplace=True)
    slices_data[s] = d
del(s, d, sm)


#%% TICKER DATA

tickers_data = {}

for sl in slices_ind:
    print('\n')
    print(sl,'...')
    
    df = slices_data[sl]
    cols = list(df.columns)
    
    for ticker in slices_tickers[sl]:
        print('     ',ticker)
        
        var = []
        for c in cols:
            if ticker == c.split('.')[0]:
               var.append(c)
               
        d = df.loc[:,var]
        d.columns = [x.split('_')[1] for x in list(d.columns)]
        tickers_data[ticker] = d

del(df, cols, var, c, ticker, d, sl)

# COMPRESSED PICKLE
with open(infolder+'tickers_data'+'.lzma','wb') as file:
    cpickle.dump(tickers_data,file,compression='lzma')


#%% MARKET INDEX DATA

data = pd.read_excel(infolder+'sp500_market_data.xlsx',sheet_name='market_index')
data.set_index('Date',inplace=True,drop=True)

# COMPRESSED PICKLE
with open(infolder+'marketindex_data'+'.lzma','wb') as file:
    cpickle.dump(data,file,compression='lzma')


#%% FIXED INCOME

data = pd.read_excel(infolder+'sp500_market_data.xlsx',sheet_name='fixed_income')
data.set_index('Date',inplace=True,drop=True)

bonds = list(data.columns)
bonds = [x.split('=')[0] for x in bonds]
bonds = list(set(bonds))

fixed_data = {}

b = bonds[0]

for b in bonds:
    print(b,'...')
    cols = list(data.columns)
    var = []
    for c in cols:
        if c.split('=')[0]==b:
           var.append(c)
    d = data.loc[:,var]
    d.columns = [x.split('_')[1] for x in list(d.columns)]
    d.dropna(inplace=True,axis=0)
    fixed_data[b] = d
    
del(b, cols, var, c, d)

# COMPRESSED PICKLE
with open(infolder+'fixedincome_data'+'.lzma','wb') as file:
    cpickle.dump(data,file,compression='lzma')


#%%


    


