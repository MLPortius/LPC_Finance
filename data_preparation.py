# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 23:47:16 2023

@author: Andriu
"""

import pandas as pd
import compress_pickle as cpickle

infolder = 'input/'
outfolder = 'output/'
plots = outfolder+'plots/'

#%% INITIAL DATA
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

data = pd.read_excel(infolder+'excel/'+'sp500_market_data.xlsx',sheet_name='fixed_income')
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
    cpickle.dump(fixed_data,file,compression='lzma')


#%% TICKER LIST

with open(infolder+'cpickle/'+'tickers_data.lzma','rb') as file:
    data_dict = cpickle.load(file)

tickers = list(data_dict.keys())
tickers.sort()

# COMPRESSED PICKLE
with open(infolder+'ticker_list'+'.lzma','wb') as file:
    cpickle.dump(tickers,file,compression='lzma')


#%% ORGANIZE DATAFRAME

# Load data
with open(infolder+'cpickle/'+'tickers_data.lzma','rb') as file:
    data_dict = cpickle.load(file)

with open(infolder+'cpickle/'+'ticker_list.lzma','rb') as file:
    ticker_list = cpickle.load(file)

del(file)

# Ordered data
filtered_data = {}
for t in ticker_list:
    d = data_dict[t]
    filtered_data[t] = {'close':d['CLOSE'],'vol':d['VOLUME']}

del(t, d)

# Joining data
close = pd.DataFrame(filtered_data[ticker_list[0]]['close'])
vols = pd.DataFrame(filtered_data[ticker_list[0]]['vol'])

for t in ticker_list[1:]:
    
    p = filtered_data[t]['close']
    v = filtered_data[t]['vol']
    
    close = pd.concat([close,p],axis=1)
    vols = pd.concat([vols,v],axis=1)
    
close.columns = ticker_list    
vols.columns = ticker_list

del(p, v, t)
del(data_dict, filtered_data)
    
#%% FILTERING DATA

import numpy as np

mv = close.isna().sum(axis=0)
mv.sort_values(ascending=False,inplace=True)

nmv = mv[mv==0]
filtered_list = list(nmv.index)
filtered_list.sort()

percentage = np.round(len(nmv)/len(mv)*100,2)

# import plotly.express as px
# from plotly.offline import plot as offplot
# from IPython.display import Image

# figsize=(2,1)
# dpi = 720
# fig = px.bar(mv2, title = 'SP500 - Close Price - Missing Values',width=figsize[0]*dpi,height=figsize[1]*dpi)
# fig.update_layout(plot_bgcolor='white',showlegend=False)
# fig.update_xaxes(showline=False,linecolor='black',showgrid=False,gridcolor='lightgrey',griddash='dash',title='Stock')
# fig.update_yaxes(showline=False,linecolor='black',showgrid=True,gridcolor='lightgrey',griddash='dash',title='Missing Values')
# offplot(fig)

# fig.write_image(plots+'missingvalues'+'.png')
# fig.write_image(plots+'missingvalues'+'.svg')
# fig.write_html(plots+'missingvalues'+'.html')

# # bytes
# img_bytes = fig.to_image(format="png")
# Image(img_bytes)

close_filtered = close.loc[:,filtered_list]
vols_filtered = vols.loc[:,filtered_list]


#%% SPLITTING DATA

def splitter(df, ratio=0.5):

    N = len(df.index)
    n = int(np.round(N*0.5,0))
    
    s1 = df.iloc[:n,:]
    s2 = df.iloc[n:,:]
    
    return s1, s2

close_f1, close_f2 = splitter(close_filtered,ratio=0.5)
vols_f1, vols_f2 = splitter(vols_filtered, ratio=0.5)

def slicer(df, ns=10):
    
    N = len(df.columns)
    n = int(np.round(N/ns,0))
    
    slices = []
    for i in range(ns):
        print('slice',i+1,'...')
        s = df.iloc[:,i*n:(i+1)*n]
        slices.append(s)
    
    return slices

slices_cf1 = slicer(close_f1, ns=10)
slices_cf2 = slicer(close_f2, ns=10)
slices_vf1 = slicer(vols_f1, ns=10)
slices_vf2 = slicer(vols_f2, ns=10)

#%% JOIN AND SAVE DATA

dataset1 = []
for i in range(len(slices_cf1)):
    dataset1.append({'close':slices_cf1[i],'vols':slices_vf1[i]}) 
    
dataset2 = []
for i in range(len(slices_cf2)):
    dataset2.append({'close':slices_cf2[i],'vols':slices_vf2[i]}) 

data = {'set1':dataset1, 'set2':dataset2}

stickers = []
for s in slices_cf1:
    stickers.append(list(s.columns))
    

# COMPRESSED PICKLE

with open(infolder+'lpc_datasets'+'.lzma','wb') as file:
    cpickle.dump(data,file,compression='lzma')
    
with open(infolder+'slice_tickers'+'.lzma','wb') as file:
    cpickle.dump(stickers,file,compression='lzma')