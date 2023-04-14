# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 20:49:55 2023

@author: Andriu
"""

#%% IMPORT LIBRARIES

import time
import eikon
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from ta.trend import ADXIndicator
import compress_pickle as cpickle
import refinitiv.dataplatform as rdp

import warnings 
warnings.filterwarnings('ignore')


#%% SETUP SCRIPT

infolder = 'input/info/'
outfolder = 'input/eikon/' 

print('\nEikon loggin...')

# Eikon
with open(outfolder+'eikon.api','r') as file:
    api_id = file.read()
    
eikon.set_app_key(api_id)

session = rdp.open_desktop_session(api_id)

sdate = '2000-01-01'
edate = datetime.datetime.now()
edate = edate.date().isoformat()

print('     ...done!')


#%% LOAD DATA

print('\nLoading tickers...')

with open(infolder+'lpc_tickers.txt','r') as file:
    lpc = file.read()
    lpc = lpc.split(';')
    lpc = lpc[:-1]

with open(infolder+'ric_tickers.txt','r') as file:
    ric = file.read()
    ric = ric.split(';')
    ric = ric[:-1]
    ric = sorted(ric)

ric_dict = {}
for r in ric:
    ric_dict[r.split('.')[0]] = r

eikon_ric = []
for tag in lpc:
    ricto = ric_dict[tag]
    eikon_ric.append(ricto)

del(tag, ricto, r, file)

print('     ...done!')


print('\nLoading prices...')

prices = cpickle.load('input/cpickle/tickers_data.lzma')

print('     ...done!')

#%% INDUSTRY

print('\nRetrieving industry data ...')

industry = {}

#   ---------------------------------------------------------------------------

sic = eikon.get_data(eikon_ric, ['TR.SICDivision', 'TR.SICMajorGroup', 'TR.SICIndustryGroup','TR.SICIndustry'])[0]
sic.index = lpc 
cols = sic.columns[1:]

# MISSING VALUES
sic2 = sic.replace('',np.nan)
mv = sic2.isna().sum(axis=0)[1:]

# UNIQUES
uniq = {}
for c in cols:
    us = list(sic[c].unique())
    if '' in us:
        us.pop(us.index(''))
    uniq[c] = len(sic[c].unique())

uniq = pd.Series(uniq)

temp = pd.concat([uniq,mv],axis=1)
temp.columns = ['UNIQUES','MISSING']

sic.replace('',np.nan,inplace=True)

industry['sic'] = {'class':'Primary Standard Industry Classification', 'info':temp, 'data':sic}


#   ---------------------------------------------------------------------------

trbc = eikon.get_data(eikon_ric, ['TR.TRBCIndustry', 'TR.TRBCIndustryGroup', 'TR.TRBCBusinessSector','TR.TRBCEconomicSector'])[0]
trbc.index = lpc
cols = trbc.columns[1:]

# MISSING VALUES
mv = trbc.replace('',np.nan)
mv = mv.isna().sum(axis=0)[1:]

# UNIQUES
uniq = {}
for c in cols:
    us = list(trbc[c].unique())
    if '' in us:
        us.pop(us.index(''))
    uniq[c] = len(trbc[c].unique())

uniq = pd.Series(uniq)

temp = pd.concat([uniq,mv],axis=1)
temp.columns = ['UNIQUES','MISSING']

trbc.replace('',np.nan,inplace=True)

industry['trbc'] = {'class':'The refinitiv business classification', 'info':temp, 'data':trbc}


#   ---------------------------------------------------------------------------

gics = eikon.get_data(eikon_ric, ['TR.GICSSubIndustry', 'TR.GICSIndustry', 'TR.GICSIndustryGroup', 'TR.GICSSector'])[0]
gics.index = lpc
cols = gics.columns[1:]

# MISSING VALUES
mv = gics.replace('',np.nan)
mv = mv.isna().sum(axis=0)[1:]

# UNIQUES
uniq = {}
for c in cols:
    us = list(gics[c].unique())
    if '' in us:
        us.pop(us.index(''))
    uniq[c] = len(gics[c].unique())

uniq = pd.Series(uniq)

temp = pd.concat([uniq,mv],axis=1)
temp.columns = ['UNIQUES','MISSING']

gics.replace('',np.nan,inplace=True)

industry['gics'] = {'class':'Primary Global Industry Classification Standard', 'info':temp, 'data':gics}

#   ---------------------------------------------------------------------------

icb = eikon.get_data(eikon_ric, ['TR.ICBIndustry', 'TR.ICBSector'])[0]
icb.index = lpc
cols = icb.columns[1:]

# MISSING VALUES
mv = icb.replace('',np.nan)
mv = mv.isna().sum(axis=0)[1:]

# UNIQUES
uniq = {}
for c in cols:
    us = list(icb[c].unique())
    if '' in us:
        us.pop(us.index(''))
    uniq[c] = len(icb[c].unique())

uniq = pd.Series(uniq)

temp = pd.concat([uniq,mv],axis=1)
temp.columns = ['UNIQUES','MISSING']

icb.replace('',np.nan,inplace=True)

industry['icb'] = {'class':'Industry Classification Benchmark', 'info':temp, 'data':icb}

#   ---------------------------------------------------------------------------

print('     ...done!')

print('\nSaving industry data ...')

with open(outfolder + 'industry_dict.lzma','wb') as file:
    cpickle.dump(industry, file, compression='lzma')

print('     ...done!')


#%% BID-ASK SPREAD

print('\nRetrieving bid ask spread ...')

bidask = {}

n = len(eikon_ric)
for i in range(n):
    tag = eikon_ric[i]
    p = np.round(((i+1)/n)*100,2)
    print('\n',tag,' - ',p,'%')
    
    data = rdp.get_historical_price_summaries(universe=tag, interval=rdp.Intervals.DAILY,
                                              start=sdate, end=edate, fields=['BID','ASK'])
    
    if type(data) == pd.DataFrame:
        data = data.astype(float)
        data['BIDASK_SPREAD'] = data['ASK']-data['BID']  
    
    bidask[tag.split('.')[0]] = data

print('\n     ...done!')


print('\nSaving bid ask data ...')

with open(outfolder + 'bidask_dict.lzma','wb') as file:
    cpickle.dump(bidask, file, compression='lzma')

print('     ...done!')


#%% ADX

print('\nRetrieving adx data ...')

adxs = {}

n = len(eikon_ric)

for i in range(n):
    
    tag = lpc[i]
    p = np.round(((i+1)/n)*100,2)
    print('\n',tag,' - ',p,'%')
    
    data = prices[tag]
    
    adx9 = ADXIndicator(high=data.loc[:,'HIGH'], low=data.loc[:,'LOW'], close=data.loc[:,'CLOSE'],window=9).adx()
    adx14 = ADXIndicator(high=data.loc[:,'HIGH'], low=data.loc[:,'LOW'], close=data.loc[:,'CLOSE'],window=14).adx()
    adx30 = ADXIndicator(high=data.loc[:,'HIGH'], low=data.loc[:,'LOW'], close=data.loc[:,'CLOSE'],window=30).adx()
    
    ADX = pd.concat([adx9, adx14, adx30], axis=1)
    ADX.columns = ['ADX9D','ADX14D','ADX30D']

    adxs[tag] = ADX

print('\n     ... done!')


print('\nSaving adx data ...')

with open(outfolder + 'adx_dict.lzma','wb') as file:
    cpickle.dump(adxs, file, compression='lzma')

print('     ...done!')


#%% VIX

print('\nRetrieving VIX data ...')

vix = yf.Ticker('^VIX')
data = vix.history(start=sdate, end=edate)
data = data.iloc[:,:4]
data.columns = [x.upper() for x in data.columns]

print('     ...done!')

print('\nSaving adx data ...')


with open(outfolder + 'vix.lzma','wb') as file:
    cpickle.dump(data, file, compression='lzma')
    
print('     ...done!')


#%% MARKET CAP

print('\nRetrieving MARKET CAP data ...')

mcaps = {}

n = len(eikon_ric)

for i in range(n):
    
    tag = eikon_ric[i]
    p = np.round(((i+1)/n)*100,2)
    print('\n',tag,' - ',p,'%')
    
    data = eikon.get_data(tag, fields=['TR.CompanyMarketCap','TR.CompanyMarketCap.date'],parameters={"SDate":sdate, "EDate":edate})

    if type(data) == tuple:
        
        data = data[0]
        
        if type(data) == pd.DataFrame:
            data = data.iloc[:,1:]
            data['Date'] = [x.split('T')[0] for x in data['Date']] 
            data['Date'] = [pd.to_datetime(x) for x in data['Date']] 
            data.set_index('Date',inplace=True,drop=True)
            data = data.astype(float)
            data.columns = ['MARKET_CAP']

            mcaps[tag.split('.')[0]] = data
            
        else:
            
            mcaps[tag.split('.')[0]] = None
            
print('\n     ...done!')


print('\nSaving MARKET CAP data ...')

with open(outfolder + 'marketcap_dict.lzma','wb') as file:
    cpickle.dump(mcaps, file, compression='lzma')

print('     ...done!')


#%% PRICE TO BOOK


print('\nRetrieving PRICE TO BOOK data ...')

n = len(eikon_ric)
f = int(n/10)

listo = []

for i in range(11):
    
    rics = eikon_ric[f*i:f*(i+1)]
    p = np.round(((i+1)/11)*100,2) 
    
    print('\n',p,'%')
    
    try:
        data = eikon.get_data(rics, fields=['TR.PriceToBVPerShare','TR.PriceToBVPerShare.date'],parameters={"SDate":sdate, "EDate":edate})
    except eikon.EikonError:
        time.sleep(30)
        data = eikon.get_data(rics, fields=['TR.PriceToBVPerShare','TR.PriceToBVPerShare.date'],parameters={"SDate":sdate, "EDate":edate})
        
    listo.append(data)
    

data = listo[0][0]
for l in listo[1:]:
    data = pd.concat([data, l[0]],axis=0)
    
    
ptbs = {}

for i in range(n):
    
    tag = eikon_ric[i]
    p = np.round(((i+1)/n)*100,2) 
    
    print('\n',tag,' - ',p,'%')
    
    d = data[data['Instrument']==tag]
    d = d.iloc[:,1:]
    d.dropna(axis=0, inplace=True)
    d['Date'] = [x.split('T')[0] for x in d['Date']] 
    d['Date'] = [pd.to_datetime(x) for x in d['Date']] 
    d.set_index('Date',inplace=True,drop=True)
    d = d.astype(float)
    d.columns = ['PBPERSHARE']
    
    ptbs[tag.split('.')[0]] = d
    
print('\n     ...done!')


print('\nSaving PRICE TO BOOK data ...')

with open(outfolder + 'pricetobook_dict.lzma','wb') as file:
    cpickle.dump(ptbs, file, compression='lzma')

print('     ...done!')


#%% ANALYSTS

print('\nRetrieving ANALYSTS EPS data ...')

n = len(eikon_ric)
f = int(n/19)

listo = []

for i in range(20):
    
    rics = eikon_ric[f*i:f*(i+1)]
    p = np.round(((i+1)/20)*100,2) 
    
    print('\n',p,'%')
    
    try:
        data, err = eikon.get_data(rics, fields=['TR.EPSMean.date','TR.EPSMean','TR.EPSStdDev',
                                                 'TR.EPSNumberofEstimates','TR.EPSNumIncEstimates'],
                                   parameters={"SDate":sdate, "EDate":edate})
    except eikon.EikonError:
        time.sleep(30)
        data, err = eikon.get_data(rics, fields=['TR.EPSMean.date','TR.EPSMean','TR.EPSStdDev',
                                                 'TR.EPSNumberofEstimates','TR.EPSNumIncEstimates'],
                                   parameters={"SDate":sdate, "EDate":edate})
    
    listo.append(data)
    
    
data = listo[0]
for l in listo[1:]:
    data = pd.concat([data, l],axis=0)
    

analysts = {}

for i in range(n):
    
    tag = eikon_ric[i]
    p = np.round(((i+1)/n)*100,2) 
    
    print('\n',tag,' - ',p,'%')
    
    d = data[data['Instrument']==tag]
    d = d.iloc[:,1:]
    d.dropna(axis=0, inplace=True)
    d['Date'] = [x.split('T')[0] for x in d['Date']] 
    d['Date'] = [pd.to_datetime(x) for x in d['Date']] 
    d.set_index('Date',inplace=True,drop=True)
    d = d.astype(float)
    d.columns = ['EST_EPS_MEAN','EST_EPS_STD','EST_EPS_NUMBER','EST_EPS_INCLUDEDNUMBER']
    d['EST_EPS_CV'] = d['EST_EPS_STD']/d['EST_EPS_MEAN']
    
    analysts[tag.split('.')[0]] = d
    
print('\n     ...done!')


print('\nSaving ANALYST EPS data ...')

with open(outfolder + 'analyst_eps_dict.lzma','wb') as file:
    cpickle.dump(analysts, file, compression='lzma')

print('     ...done!')


#%% RECOMENDATIONS

print('\nRetrieving RECOMENDATIONS data ...')

n = len(eikon_ric)
f = int(n/19)

listo = []

for i in range(20):
    
    rics = eikon_ric[f*i:f*(i+1)]
    p = np.round(((i+1)/20)*100,2) 
    
    print('\n',p,'%')
    
    try:
        data, err = eikon.get_data(rics, fields=['TR.NumOfStrongBuy.date','TR.NumOfStrongBuy','TR.NumOfBuy',
                                                 'TR.NumOfHold','TR.NumOfSell','TR.NumOfStrongSell'],
                                   parameters={"SDate":sdate, "EDate":edate})
    except eikon.EikonError:
        time.sleep(30)
        data, err = eikon.get_data(rics, fields=['TR.NumOfStrongBuy.date','TR.NumOfStrongBuy','TR.NumOfBuy',
                                                 'TR.NumOfHold','TR.NumOfSell','TR.NumOfStrongSell'],
                                   parameters={"SDate":sdate, "EDate":edate})
    listo.append(data)
    
    
data = listo[0]
for l in listo[1:]:
    data = pd.concat([data, l],axis=0)
    
    
recom = {}

for i in range(n):
    
    tag = eikon_ric[i]
    p = np.round(((i+1)/n)*100,2) 
    
    print('\n',tag,' - ',p,'%')
    
    d = data[data['Instrument']==tag]
    d = d.iloc[:,1:]
    d.dropna(axis=0, inplace=True)
    d['Date'] = [x.split('T')[0] for x in d['Date']] 
    d['Date'] = [pd.to_datetime(x) for x in d['Date']] 
    d.set_index('Date',inplace=True,drop=True)
    d = d.astype(int)
    d.columns = ['STRONGBUY','BUY','HOLD','SELL','STRONGSELL']
    
    recom[tag.split('.')[0]] = d
    
print('\n     ...done!')


print('\nSaving RECOMMENDATIONS data ...')

with open(outfolder + 'recommend_dict.lzma','wb') as file:
    cpickle.dump(recom, file, compression='lzma')

print('     ...done!')


#%% FIXED INCOME

print('\nLoading FIXED INCOME data ...')

fixed = cpickle.load('input/cpickle/fixedincome_data.lzma')

m1 = fixed['US1MT'].loc[:,'CLOSE']
y30 = fixed['US30YT'].loc[:,'CLOSE']

data = pd.concat([y30,m1], axis=1)
data.columns = ['US30YT','US1MT']
data['BOND_SPREAD'] = data['US30YT']-data['US1MT']

print('     ...done!')

print('\nSaving BOND SPREAD...')

with open(outfolder + 'us_bondspread_dict.lzma','wb') as file:
    cpickle.dump(data, file, compression='lzma')

print('     ...done!')


#%% VOLATILITY

print('\nPreparing VOLATILITY data ...')

data = {}

for tag in lpc:
    
    d = prices[tag]
    d = d['CLOSE'].to_frame()
    d['Ri'] = (d['CLOSE'] - d['CLOSE'].shift(1)) / d['CLOSE'].shift(1)
    d['Ri'] = d['Ri'] * 100
    d.dropna(axis=0, inplace=True)
    data[tag] = d

stds = {}

for tag in lpc:
    
    d = data[tag]
    
    d['P_STD_21'] = d['CLOSE'].rolling(21).std()
    d['R_STD_21'] = d['Ri'].rolling(21).std()
    
    d['P_STD_252'] = d['CLOSE'].rolling(252).std()
    d['R_STD_252'] = d['Ri'].rolling(252).std()
    
    stds[tag] = d

print('\n     ...done!')


print('\nSaving VOLATILITY...')

with open(outfolder + 'volatility_dict.lzma','wb') as file:
    cpickle.dump(stds, file, compression='lzma')

print('     ...done!')
    

#%% HURST EXPONENT

print('\nPreparing HURST data ...')

set1_dates = ['','']
set2_dates = ['','']
set3_dates = ['','']


def GET_HURST(serie, max_lag=21):
        
    lags = list(range(2, max_lag))
    
    tau = []
    for l in lags:
        _ = serie - serie.shift(l)
        _.dropna(axis=0, inplace=True)
        tau.append(_.std())
        
    tau = np.asarray(tau)
    lags = np.asarray(lags)
    
    reg = np.polyfit(np.log(lags), np.log(tau), 1)

    return reg[0]


def GET_HURST_GRIDMEAN(time_serie):
    
    # 1TM, 2TM, 3TM, 5TM, 7TM, 9TM, 1TY, 1.2TY, 1.5TY, 1.7TY, 2TY 
    grid = [21, 42, 63, 147, 189, 252, 315, 378, 441, 504]
    
    hs = []
    for mlag in grid:
        hs.append(GET_HURST(time_serie, max_lag = mlag))
        
    hs = pd.Series(hs)
    hsm = hs.mean()
    
    return hsm


def GET_HURST_CLASS(x):
    
    x2 = np.round(x, 1)
    
    if x2 == 0.5:
        y = 'RANDWALK'
        
    elif x2 < 0.5:
        y = 'MEANREV'
        
    elif x2 > 0.5:
        y = 'TRENDING'
        
    else:
        y = 'ERROR'
    
    return y 

    
hursts = {}
    
for tag in lpc:

    print(tag, '...')
    d = prices[tag]['CLOSE'].copy()
    
    # METRIC 1 - GRIDMEAN
    hgm = GET_HURST_GRIDMEAN(d)
        
    # METRIC 2 - GRIDMEAN CLASS
    hgmc = GET_HURST_CLASS(hgm)

    # METRIC 3 - ROLLING MEAN
    r = d.rolling(252).apply(GET_HURST)
    r = r.to_frame()
    r.index = d.index
    r.columns = ['HURST']
    r = pd.Series(r['HURST'])
    r.dropna(inplace=True)

    hrmean = r.mean()
    hrlast = r[-1]

    # METRIC 4 - ROLLING CLASS PROBA
    r = r.to_frame()
    r['CAT'] = r['HURST'].apply(GET_HURST_CLASS)

    dum = pd.get_dummies(r['CAT'], prefix='HURST',dtype=int)

    probas = {'rw':dum['HURST_RANDWALK'].mean(), 'mr':dum['HURST_MEANREV'].mean(), 'tr':dum['HURST_TRENDING'].mean()}

    # METRIC 5 - ROLLING CLASS DAYS
    days = {'rw':dum['HURST_RANDWALK'].sum(), 'mr':dum['HURST_MEANREV'].sum(), 'tr':dum['HURST_TRENDING'].sum()}

    # METRIC 6 - DISTANCE FROM RANDOM WALK
    r['DFRW'] = np.abs(r['HURST'] - 0.5)

    dfrw = r['DFRW'].mean()

    dicto = {'hgm':hgm, 'hgmc':hgmc, 'rdf':r, 'rmean':hrmean,'rlast':hrlast, 'pr':probas, 'dy':days, 'dfrw':dfrw}
    
    hursts[tag] = dicto

hursts_sets = {}
hursts_sets['set3'] = hursts


