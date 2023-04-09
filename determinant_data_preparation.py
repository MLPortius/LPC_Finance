# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 10:26:31 2023

@author: Andriu
"""

#%% IMPORT LIBRARIES

import os
import argparse
import numpy as np
import pandas as pd
import compress_pickle as cpickle


#%% SCRIPT SETUP

# parser = argparse.ArgumentParser(prog= 'LPC_Determinant_Analysis_Data_Preparation',
#                                  description = 'Recopiles LPC data and creates dataframes',
#                                  epilog = 'Created by Andriu')


# parser.add_argument('-d','--dset', required=True, type=int, choices=[1,2,3])
# args = parser.parse_args()

# # ARGUMENTS

# ds = args.dset


#%% ENVIRONMENT SETUP

ds = 1

infolder = 'output/analysis/'
infolder2 = 'input/eikon/'
infolder3 = 'input/cpickle/'

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

def DIR(x):
    if x >= 0:
        y = 1
    else:
        y = 0
    return y
    

def GET_RETURNS_DICT(i, save=False):
    
    dicto = cpickle.load('input/cpickle/lpc_datasets.lzma')

    d1 = dicto['set1']
    d2 = dicto['set2']
    
    d1 = [x['close'] for x in d1]
    d2 = [x['close'] for x in d2]
    
    df1 = d1[0]
    for d in d1[1:]:
        df1 = pd.concat([df1, d], axis=1)
    
    df2 = d2[0]
    for d in d2[1:]:
        df2 = pd.concat([df2, d], axis=1)
    
    df3 = pd.concat([df1, df2], axis=0)
    
    if save:
        dts = {'set1':df1.index, 'set2':df2.index, 'set3':df3.index}
        with open('input/info/datasets_dates.lzma', 'wb') as file:
            cpickle.dump(dts, file, compression='lzma')
    
    else:
        
        if i == 1:
            output = {}
            for c in list(df1.columns):
                output[c] = df1.loc[:,c]
        
        elif i == 2:
            output = {}
            for c in list(df2.columns):
                output[c] = df2.loc[:,c]
        
        elif i == 3:
            output = {}
            for c in list(df3.columns):
                output[c] = df3.loc[:,c]
                
        else:
            output = 'error'
            
        return output
        
    
#%% TICKERS 

# GET_RETURNS_DICT(i=3, save=True)

print('\nLoading tickers...')

with open('input/info/lpc_tickers.txt','r') as file:
    tickers = file.read()
    tickers = tickers.split(';')
    tickers = tickers[:-1]

del(file)

print('     ...done!')


print('\nLoading dates...')

dates = cpickle.load('input/info/datasets_dates.lzma')
dates = dates[dset]

print('     ...done!')

main = pd.DataFrame(index=tickers)


#%% DEPVAR

print('\nPreparing depvars...')

# ----------------------------------- DA --------------------------------------

print('     ... DA')

data = cpickle.load('output/trading_simulation/'+dset+'/da_grid_results.lzma')

das = {}

for t in tickers:
    
    d = data[t]

    d['SIGNAL'] = d['NORM_SIGNAL'] * d['STD'] + d['MEANS']
    d['PREDICTION'] = d['NORM_PREDICTION'] * d['STD'] + d['MEANS']
    
    dn = d.loc[:,['NORM_SIGNAL','NORM_PREDICTION']]
    dnn = d.loc[:,['SIGNAL','PREDICTION']]
    
    dn['DT'] = dn['NORM_SIGNAL'] - dn['NORM_SIGNAL'].shift(1)
    dn['DP'] = dn['NORM_PREDICTION'] - dn['NORM_PREDICTION'].shift(1)
    
    dn.dropna(axis=0, inplace=True)
    
    dn['COMPARE'] = dn['DT'] * dn['DP']
    dn['HIT'] =  dn['COMPARE'].apply(DIR)
    
    da_n = dn['HIT'].mean()
    
    dnn['DT'] = dnn['SIGNAL'] - dnn['SIGNAL'].shift(1)
    dnn['DP'] = dnn['PREDICTION'] - dnn['PREDICTION'].shift(1)
    
    dnn.dropna(axis=0, inplace=True)
    
    dnn['COMPARE'] = dnn['DT'] * dnn['DP']
    dnn['HIT'] =  dnn['COMPARE'].apply(DIR)
    
    da_nn = dnn['HIT'].mean()

    das[t] = [da_nn, da_n]

das = pd.DataFrame(das.values(), index=das.keys())
das.columns = ['OPT_DA','OPT_NORM_DA']

# dt = pd.merge(dnn, dn, right_index=True, left_index=True, how='inner')
# dt['COMPARES'] = dt['HIT_x'] - dt['HIT_y']

das = das.loc[:,'OPT_DA']
das = das * 100

main = pd.concat([main, das],axis=1)


# ----------------------------------- DIS -------------------------------------

print('     ... DIS')

data = cpickle.load('output/trading_simulation/'+dset+'/dis_grid_results.lzma')

diss = {}

for t in tickers:
    
    d = data[t]

    d['NORM_ERROR'] = d['NORM_PREDICTION'] - d['NORM_SIGNAL']
    d['SIGNAL'] = d['NORM_SIGNAL'] * d['STD'] + d['MEANS']
    d['PREDICTION'] = d['NORM_PREDICTION'] * d['STD'] + d['MEANS']
    d['ERROR'] = d['PREDICTION'] - d['SIGNAL']

    signal = list(d['SIGNAL'])
    error = list(d['ERROR'])
    norm_signal = list(d['NORM_SIGNAL'])
    norm_error = list(d['NORM_ERROR'])

    tau = 0
    
    xs_s = []
    xs_e = []
    xs_ns = []
    xs_ne = []

    for i in range(len(signal)-tau):
        
        x1 = signal[i+tau] * signal[i]
        x2 = error[i+tau] * error[i]
        x3 = norm_signal[i+tau] * norm_signal[i]
        x4 = norm_error[i+tau] * norm_error[i]
        
        xs_s.append(x1)
        xs_e.append(x2)
        xs_ns.append(x3)
        xs_ne.append(x4)
    
    ac_s = sum(xs_s) / (len(signal) - tau)
    ac_e = sum(xs_e) / (len(signal) - tau)
    ac_ns = sum(xs_ns) / (len(signal) - tau)         
    ac_ne = sum(xs_ne) / (len(signal) - tau)
    
    dis = ac_e/ac_s
    dis_n = ac_ne/ac_ns

    diss[t] = [dis, dis_n]

diss = pd.DataFrame(diss.values(), index=diss.keys())
diss.columns = ['OPT_DIS','OPT_NORM_DIS']

diss = diss.loc[:,'OPT_DIS']
diss = diss * 100

main = pd.concat([main, diss],axis=1)


# ----------------------------------- MAPE ------------------------------------

print('     ... MAPE')

data = cpickle.load('output/trading_simulation/'+dset+'/mae_grid_results.lzma')

mapes = {}

for t in tickers:
    
    d = data[t].copy()
    d['SIGNAL'] = d['NORM_SIGNAL'] * d['STD'] + d['MEANS']
    d['PREDICTION'] = d['NORM_PREDICTION'] * d['STD'] + d['MEANS']
    
    d['E_NN'] = d['PREDICTION'] - d['SIGNAL']
    d['PE_NN'] = d['E_NN']/d['SIGNAL']
    d['APE_NN'] = d['PE_NN'].abs()
    mape_nn = d['APE_NN'].mean()
    
    d['AE_NN'] = d['E_NN'].abs()
    mae_nn = d['AE_NN'].mean()
    
    d['E_N'] = d['NORM_PREDICTION'] - d['NORM_SIGNAL']
    d['PE_N'] = d['E_N']/d['NORM_SIGNAL']
    d['APE_N'] = d['PE_N'].abs()
    mape_n = d['APE_N'].mean()
    
    d['AE_N'] = d['E_N'].abs()
    mae_n = d['AE_N'].mean()
    
    mapes[t] = [mae_nn, mape_nn, mae_n, mape_n]

mapes = pd.DataFrame(mapes.values(), index=mapes.keys(),
                     columns=['OPT_MAE','OPT_MAPE','OPT_MAE_NORM','OPT_MAPE_NORM'])

mapes = mapes.loc[:,'OPT_MAPE']
mapes = mapes * 100

main = pd.concat([main, mapes], axis=1)

print('     ...done!')

del(ac_e, ac_ne, ac_ns, ac_s, d, da_n, da_nn, das, data, dis, dis_n, diss, dn, dnn, error, i, mae_n, mae_nn, mape_n, mape_nn, mapes, norm_error, norm_signal, signal, t, tau, x1, x2, x3, x4, xs_ns, xs_e, xs_ne, xs_s)


#%% FIXED INCOME

print('\nPreparing fixed income data..')

spread = cpickle.load(infolder2+'us_bondspread_dict.lzma')

data = GET_RETURNS_DICT(i=ds)
 
for t in tickers:
    d = data[t].to_frame()
    d['RETURN'] = (d[t] - d[t].shift(1))/d[t].shift(1) 
    d.dropna(axis=0, inplace=True)
    data[t] = d['RETURN'] * 100

fixed = {}

for t in tickers:
    
    d = data[t]
    d = pd.merge(d, spread, left_index=True, right_index=True, how='left')
    d = d.iloc[:,[0,3]]
    d.columns = ['RETURN','SPREAD']
    d.ffill(inplace=True)

    cov = d.cov()
    b = cov.loc['RETURN','SPREAD']/cov.loc['SPREAD','SPREAD']
    
    fixed[t] = b

fixed = pd.DataFrame(fixed.values(), index=fixed.keys())
fixed.columns = ['SPREAD_BETA']

main = pd.concat([main, fixed], axis=1)

del(spread, d, data, cov, b, fixed, t)

print('     ...done!')


#%% MARKET BETA

print('\nPreparing market beta data..')

market = cpickle.load(infolder3+'marketindex_data.lzma')
market.columns = ['SP500']
market['Rm'] = (market['SP500']-market['SP500'].shift(1))/market['SP500'].shift(1)
market['Rm'] = market['Rm']*100 
market.dropna(axis=0, inplace=True)

data = GET_RETURNS_DICT(i=ds)
 
for t in tickers:
    d = data[t].to_frame()
    d['Ri'] = (d[t] - d[t].shift(1))/d[t].shift(1) 
    d.dropna(axis=0, inplace=True)
    data[t] = d['Ri'] * 100

mindex = {}

for t in tickers:
    
    d = data[t]
    d = pd.merge(d, market, left_index=True, right_index=True, how='left')
    d = d.iloc[:,[0,2]]
    d.columns = ['Ri','Rm']
    d.ffill(inplace=True)

    cov = d.cov()
    b = cov.loc['Ri','Rm']/cov.loc['Rm','Rm']
    
    mindex[t] = b
 
mindex = pd.DataFrame(mindex.values(), index=mindex.keys())
mindex.columns = ['MARKET_BETA']

main = pd.concat([main, mindex], axis=1)

del(market, d, data, cov, b, mindex, t)

print('     ...done!')


#%% VOLATILITY

print('\nPreparing volatility data...')

data = cpickle.load(infolder2 + 'volatility_dict.lzma')

for t in tickers:

    if ds == 1:
        data[t] = data[t].loc[dates[1:], :]    
    
    elif ds == 2:
        data[t] = data[t].loc[dates, :]    
    
    elif ds == 3:
        data[t] = data[t].loc[dates[1:], :]   
        
volas = {}

for t in tickers:
    
    d = data[t]
    
    stv = d['R_STD_21'].mean()
    ltv = d['R_STD_252'].mean()

    volas[t] = [stv, ltv]

volas = pd.DataFrame(volas.values(), index=volas.keys())
volas.columns = ['R_MSTD_ST', 'R_MSTD_LT']

main = pd.concat([main, volas], axis=1)

print('     ...done!')

del(volas, t, stv, ltv, data, d)


#%% LPC

print('\nPreparing LPC data...')

da = pd.read_excel(infolder+dpath+'da_opt.xlsx', index_col='stock')
da = da.loc[:,['w_size', 'p_lags']]
da.columns = ['DA_W_SIZE', 'DA_P_LAGS']

dis = pd.read_excel(infolder+dpath+'dis_opt.xlsx', index_col='stock')
dis = dis.loc[:,['w_size', 'p_lags']]
dis.columns = ['DIS_W_SIZE', 'DIS_P_LAGS']

mae = pd.read_excel(infolder+dpath+'mae_opt.xlsx', index_col='stock')
mae = mae.loc[:,['w_size', 'p_lags']]
mae.columns = ['MAPE_W_SIZE', 'MAPE_P_LAGS']

main = pd.concat([main, da, dis, mae], axis=1)

print('     ...done!')

del(da, dis, mae)


#%%

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
