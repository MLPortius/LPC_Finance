# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 10:26:31 2023

@author: Andriu
"""

#%% IMPORT LIBRARIES

import argparse
import numpy as np
import pandas as pd
import compress_pickle as cpickle


#%% SCRIPT SETUP

parser = argparse.ArgumentParser(prog= 'LPC_Determinant_Analysis_Data_Preparation',
                                  description = 'Recopiles LPC data and creates dataframes',
                                  epilog = 'Created by Andriu')


parser.add_argument('-d','--dset', required=True, type=int, choices=[1,2,3])
args = parser.parse_args()

# ARGUMENTS

ds = args.dset


#%% ENVIRONMENT SETUP

infolder = 'output/analysis/'
infolder2 = 'input/eikon/'
infolder3 = 'input/cpickle/'

outfolder = 'output/determinants/'

dset = 'set'+str(ds)
dpath = dset+'/' 


#%% FUNCTION DEFINITION

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
        
def GET_VOLUMES_DICT(i):
    
    dicto = cpickle.load('input/cpickle/lpc_datasets.lzma')
    
    d1 = dicto['set1']
    d2 = dicto['set2']

    d1 = [x['vols'] for x in d1]
    d2 = [x['vols'] for x in d2]
    
    df1 = d1[0]
    for d in d1[1:]:
        df1 = pd.concat([df1, d], axis=1)
    
    df2 = d2[0]
    for d in d2[1:]:
        df2 = pd.concat([df2, d], axis=1)
        
    df3 = pd.concat([df1, df2], axis=0)
    
    
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

# https://www.avatrade.es/educacion/trading-para-principiantes/indicador-adx
# https://www.investopedia.com/articles/trading/07/adx-trend-indicator.asp

def ADX_CAT(x):
    
    if x < 25:
        y = 'WEAK_NO_TREND'
    elif x >= 25 and x<50:
        y = 'STRONG_TREND'
    elif x >= 50 and x<75:
        y = 'VERY_STRONG_TREND'
    elif x>=75:
        y = 'EXTREMELY_STRONG_TREND'
    
    return y

def ADX_BINARY(x):
    if x < 25:
        y = 0
    else:
        y = 1
    return y

        
    
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


#%% MARKET CAP

print('\nPreparing MARKET CAP data...')

data = cpickle.load(infolder2 + 'marketcap_dict.lzma')
close = GET_RETURNS_DICT(ds)

mcaps = {}

for t in tickers:
    
    d = data[t]
    d = d/1000000
    
    c = close[t]
    
    m = pd.merge(c, d, left_index=True, right_index=True, how='left')
    m.ffill(inplace=True)
    m = m['MARKET_CAP']
    
    mcap_mean = m.mean()
    mcap_last = m.iloc[-1]
    
    m2 = m.rolling(21).mean()
    m2.dropna(axis=0, inplace=True)
    
    mcap_rmean = m2.mean()
    
    mcaps[t] = [mcap_last, mcap_mean, mcap_rmean]

mcaps = pd.DataFrame(mcaps.values(), index=mcaps.keys())
mcaps.columns = ['MARKETCAP_LAST','MARKETCAP_MEAN', 'MARKETCAP_RMEAN']

main = pd.concat([main, mcaps['MARKETCAP_RMEAN']],axis=1)

print('     ...done!')

del(data, close, mcaps, t, d, c, m, mcap_last, mcap_mean, mcap_rmean, m2)


#%% VOLUME

print('\nPreparing VOLUMES data...')

close = GET_RETURNS_DICT(ds)
vols = GET_VOLUMES_DICT(ds)

vs = {}

for t in tickers:
    
    c = close[t]
    r = (c - c.shift(1))/c.shift(1)
    
    v = vols[t]
    v = v/1000000
    
    v_mean = v.mean()
    v_std = v.std()
    v_cv = (v_std/v_mean)*100
    
    v2 = v.rolling(21).mean()
    v2.dropna(inplace=True)
    
    v_rmean = v2.mean()
    
    v_last = v.iloc[-1]
    
    d = pd.concat([v, r], axis=1)
    d.dropna(inplace=True)
    d.columns = ['Vi','Ri']
    
    v_rcorr = np.corrcoef(d['Vi'],d['Ri'])[1][0]
    
    cov = d.cov()
    v_rbeta = cov.loc['Vi','Ri']/cov.loc['Vi','Vi']
    
    vs[t] = [v_last, v_mean, v_cv, v_rmean, v_rcorr, v_rbeta]

vs = pd.DataFrame(vs.values(), index=vs.keys())

vs.columns = ['VOLUME_LAST', 'VOLUME_MEAN', 'VOLUME_CV', 
              'VOLUME_RMEAN', 'VOLUME_RET_CORR', 'VOLUME_RET_BETA']

vs = vs.loc[:,'VOLUME_CV']

main = pd.concat([main, vs], axis=1)

print('     ...done!')

del(d, c, cov, r, t, v, v2, v_cv, v_last, v_mean, v_rbeta, v_rcorr, v_rmean, v_std, vols, vs, close)


#%% PRICE TO BOOK

print('\nPreparing PRICE TO BOOK data...')

data = cpickle.load(infolder2 + 'pricetobook_dict.lzma')

close = GET_RETURNS_DICT(ds)

pbps = {}

for t in tickers:
    
    p = data[t]
    c = close[t]
    
    d = pd.merge(c, p, left_index=True, right_index=True, how='left')
    d = d['PBPERSHARE']
    
    pbps_last = d.iloc[-1]
    pbps_mean = d.mean()
    pbps_rmean = d.rolling(21).mean().mean()
    
    pbps[t] = [pbps_last, pbps_mean, pbps_rmean]   

pbps = pd.DataFrame(pbps.values(), index = pbps.keys())
pbps.columns = ['PBPERSHARE_LAST','PBPERSHARE_MEAN', 'PBPERSHARE_RMEAN']

main = pd.concat([main, pbps.loc[:,['PBPERSHARE_RMEAN']]], axis=1)

print('     ...done!')

del(c, close, d, data, p, pbps, pbps_last, pbps_mean, pbps_rmean, t)


#%% STREAK PROBA

print('\nPreparing STREAK HISTOGRAM data...')

da = pd.read_excel(infolder+dpath+'da_opt.xlsx', index_col='stock')

dh = pd.read_excel('output/analysis/'+dset+'/histos.xlsx')
cols = list(dh.columns)
cols[0] = 'stock'
dh.columns = cols
dh.set_index('stock', drop=True, inplace=True)

d = pd.concat([da['ht_mean'], dh], axis=1)
d.columns = ['STREAK_HMEAN','STREAK_PROBA','MAX_STREAK']
d['STREAK_PROBA'] = d['STREAK_PROBA'] * 100

main = pd.concat([main, d], axis=1)

print('     ...done!')

del(cols, d, da, dh)


#%% ADX

print('\nPreparing ADX data...')

data = cpickle.load(infolder2 + 'adx_dict.lzma')
close = GET_RETURNS_DICT(ds)

adxs = {}

for t in tickers:
    
    d = data[t]
    c = close[t]
    
    d = pd.merge(c, d, left_index=True, right_index=True, how='left')
    d = d['ADX30D']
    d.replace(0, np.nan, inplace=True)
    d.dropna(inplace=True)
    
    adx_mean = d.mean()
    
    df = d.to_frame()
    df.columns = ['ADX']
    df['CAT'] = df['ADX'].apply(ADX_BINARY)

    tdays = df['CAT'].sum()
    tproba = df['CAT'].mean()
    tproba = tproba * 100
    
    adxs[t] = [adx_mean, tdays, tproba]

adxs = pd.DataFrame(adxs.values(), index = adxs.keys())
adxs.columns = ['ADX30D_MEAN','ADX30D_TREND_DAYS', 'ADX30D_TREND_PROBA']

adxs = adxs.loc[:,['ADX30D_TREND_DAYS','ADX30D_TREND_PROBA']]

main = pd.concat([main, adxs], axis=1)

print('     ...done!')

del(adx_mean, adxs, c, close, d, data, df, t, tdays, tproba)


#%% VIX

print('\nPreparing VIX data...')

data = cpickle.load(infolder2 + 'vix.lzma')
vix = data['CLOSE'].to_frame()
vix.columns = ['VIXi']
vix.index = [x.date() for x in list(vix.index)]

vix['VIX_Ri'] = (vix['VIXi'] - vix['VIXi'].shift(1))/vix['VIXi'].shift(1)
vix.dropna(axis=0, inplace=True)
vix['VIX_Ri'] = vix['VIX_Ri']*100

close = GET_RETURNS_DICT(ds)

vixs = {}

for t in tickers:
    
    c = close[t]
    r = (c - c.shift(1))/c.shift(1)

    r.dropna(inplace=True)
    r = r * 100
    
    d = pd.merge(r, vix, left_index=True, right_index=True, how='left')
    d.columns = ['Ri', 'VIXi','VIXi_Ri']    
    
    cov = d.cov()

    vbeta = cov.loc['Ri','VIXi_Ri']/cov.loc['VIXi_Ri','VIXi_Ri']

    vixs[t] = vbeta
    
vixs = pd.DataFrame(vixs.values(), index = vixs.keys())
vixs.columns = ['VIX_BETA']

main = pd.concat([main, vixs], axis=1)

print('     ...done!')

del(c, close, cov, d, data, r, t, vbeta, vix, vixs)


#%% ANALYST DISPERSION

print('\nPreparing ANALYST DISPERSION data ...')

data = cpickle.load(infolder2 + 'analyst_eps_dict.lzma')
close = GET_RETURNS_DICT(ds)

adisp = {}

t = 'A'

for t in tickers:
    
    d = data[t]
    d = d['EST_EPS_CV']
    
    c = close[t]
    
    m = pd.merge(c, d, left_index=True, right_index=True, how='left')
    m.dropna(axis=0, inplace=True)

    disp = m['EST_EPS_CV'] * 100
    
    if len(disp) != 0:
 
        disp_mean = disp.mean()
        disp_last = disp[-1]
        disp_rmean = disp.rolling(21).mean().mean()
        
    else:
        
        disp_mean = np.nan
        disp_last = np.nan
        disp_rmean = np.nan
        
    adisp[t] = [disp_mean, disp_last, disp_rmean]

adisp = pd.DataFrame(adisp.values(), index=adisp.keys())
adisp.columns = ['ANALYST_DISPERSION_MEAN', 'ANALYST_DISPERSION_LAST', 'ANALYST_DISPERSION_RMEAN']

main = pd.concat([main, adisp['ANALYST_DISPERSION_RMEAN']], axis=1)

print('     ...done!')

del(data, close, adisp, t, d, c, m, disp, disp_mean, disp_last)


#%% ANALYST NUMBER

print('\nPreparing ANALYST NUMBER data ...')

data = cpickle.load(infolder2 + 'analyst_eps_dict.lzma')
close = GET_RETURNS_DICT(ds)

an = {}

for t in tickers:
    
    d = data[t]
    c = close[t]
    
    m = pd.merge(c, d['EST_EPS_NUMBER'], left_index=True, right_index=True, how='left')
    m = m['EST_EPS_NUMBER']
    m.dropna(inplace=True)
    
    if len(m) != 0:
        
        an_mean = m.mean()
        an_last = m[-1]
        an_rmean = m.rolling(21).mean().mean()

    else:
        
        an_mean = np.nan
        an_last = np.nan
        an_rmean = np.nan
        
    an[t] = [an_mean, an_last, an_rmean]
    
an = pd.DataFrame(an.values(), index = an.keys())
an.columns = ['ANALYST_NUMBER_MEAN','ANALYST_NUMBER_LAST', 'ANALYST_NUMBER_RMEAN']

main = pd.concat([main, an['ANALYST_NUMBER_RMEAN']], axis=1)

print('     ...done!')

del(data, close, an, t, d, c, m, an_mean, an_last)


#%% PUT CALL RATIO

print('\nPreparing PUT CALL RATIO data ...')

data = cpickle.load(infolder3 + 'putcallratio.lzma')
close = GET_RETURNS_DICT(ds)

pcrs = {}

for t in tickers:
    
    d = data[t]
    d = d['PC_RATIO'] 
    d.replace(np.inf, np.nan, inplace=True)
    d.ffill(inplace=True)
    
    c = close[t] 
    m = pd.merge(c, d, left_index=True, right_index=True, how='left')
    m = m['PC_RATIO']
    
    if len(m) != 0:
        
        pcr_mean = m.mean()
        pcr_last = m.iloc[-1]
        pcr_rmean = m.rolling(21).mean().mean()
    
    else:
        
        pcr_mean = np.nan
        pcr_last = np.nan
        pcr_rmean = np.nan
    
    pcrs[t] = [pcr_mean, pcr_last, pcr_rmean]
    
pcrs = pd.DataFrame(pcrs.values(), index = pcrs.keys())
pcrs.columns = ['PUTCALLRATIO_MEAN','PUTCALLRATIO_LAST','PUTCALLRATIO_RMEAN']

main = pd.concat([main, pcrs['PUTCALLRATIO_RMEAN']], axis=1)

print('     ...done!')

del(data, close, pcrs, t, d, c, m, pcr_mean, pcr_last, pcr_rmean)


#%% BIDASK

print('\nPreparing BIDASK SPREAD data ...')

data = cpickle.load(infolder2 + 'bidask_dict.lzma')
close = GET_RETURNS_DICT(ds)

bas = {}

for t in tickers:
    
    if type(data[t]) == type(None):
        
        ba_mean = np.nan
        ba_last = np.nan
        ba_rmean = np.nan
    
    else:
    
        d = data[t]['BIDASK_SPREAD']
        d.ffill(inplace=True)
        
        c = close[t]
        
        m = pd.merge(c, d, left_index=True, right_index=True, how='left')
        m = m['BIDASK_SPREAD']
        
        if len(m) != 0:
            
            ba_mean = m.mean()
            ba_last = m.iloc[-1]
            ba_rmean = m.rolling(21).mean().mean()
            
        else:
            
            ba_mean = np.nan
            ba_last = np.nan
            ba_rmean = np.nan
            
    bas[t] = [ba_mean, ba_last, ba_rmean]
    
bas = pd.DataFrame(bas.values(), index=bas.keys())
bas.columns = ['BIDASK_MEAN','BIDASK_LAST','BIDASK_RMEAN']

main = pd.concat([main, bas['BIDASK_RMEAN']], axis=1) 

print('     ...done!')

del(data, close, bas, t, d, c, m, ba_mean, ba_last, ba_rmean)


#%% CUSUM

print('\nPreparing CUSUM BREAKS data ...')

data = cpickle.load(infolder2 + 'cusum_dict.lzma')
data = data[dset]

for t in tickers:
    d = data[t]
    dls = list(d.iloc[0,:])
    data[t] = dls
    
df = pd.DataFrame(data.values(), index=data.keys())
df.columns = d.columns

main = pd.concat([main, df['PRICE_CUSUM_BREAKS']], axis=1)

print('     ...done!')

del(data, t, d, dls, df)


#%% HURST

print('\nPreparing HURST EXPONENT data ...')

data = cpickle.load(infolder2 + 'hurst_dict.lzma')
data = data[dset]

for t in tickers:
    d = data[t]['m']
    d.index = [t]
    data[t] = d

df = data[tickers[0]]
for t in tickers[1:]:
    df = pd.concat([df, data[t]], axis=0)
    
df.fillna(0, inplace=True)

df2 = df.loc[:,['HURST_RANDWALK_DAYS','HURST_RANDWALK_PROBA','HURST_MEANREV_DAYS','HURST_MEANREV_PROBA','HURST_TRENDING_DAYS','HURST_TRENDING_PROBA']]

main = pd.concat([main, df2], axis=1)

print('     ...done!')

del(data, t, d, df, df2)


#%% INDUSTRY

print('\nPreparing INDUSTRY data ...')

data = cpickle.load(infolder2+'industry_dict.lzma')

d = data['gics']['data']
d = d.loc[:,'GICS Industry Group Name']

dum = pd.get_dummies(d, prefix='INDUSTRY', dtype=int)

main = pd.concat([main, dum], axis=1)

print('     ...done!')

del(data, d, dum)


#%% EXPORT DETERMINANT DATA

print('\nExporting data...')

with open(outfolder + dpath + 'det_main_data.lzma','wb') as file:
    cpickle.dump(main, file, compression='lzma')

main.to_excel(outfolder + dpath + 'det_main_data.xlsx')

print('     ...Done!')
