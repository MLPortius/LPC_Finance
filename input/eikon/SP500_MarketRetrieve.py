# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 16:34:39 2022

@author: Andriu
"""

#%% IMPORT PACKAGES

import numpy as np
import pandas as pd

import eikon
import datetime

import sqlalchemy as sa

#%% PACKAGE CONFIGURATION

# Eikon
application_id = '1b82dc11459c4950ad6f90740e7ace636e5d9d3a' 
eikon.set_app_key(application_id)


# SQLalchemy
tipo_db='postgresql'
user='postgres'
password='187397'
host='cx360.duckdns.org'
database='SP500'

con_string=tipo_db+"://"+user+":"+password+"@"+host+"/"+database
engine = sa.create_engine(con_string)

#%% CREATE SCHEMAS

QUERY = '''CREATE SCHEMA IF NOT EXISTS "STOCKS" '''
engine.execute(QUERY)

QUERY = ''' CREATE SCHEMA IF NOT EXISTS "MARKET_INDEX" '''
engine.execute(QUERY)

QUERY = ''' CREATE SCHEMA IF NOT EXISTS "FIXED_INCOME"  '''
engine.execute(QUERY)

#%% LOAD STOCKS' INDEX

# Retrieve RICS from Composite SP500 Index
SP500_DATA, err = eikon.get_data('0#.SPXTR', "TR.RIC") 
SP500_IND_LIST = list(SP500_DATA.loc[:,'RIC'])


# Make Slices to Retrieve
SP500_IND_SLICES = []
for i in range(0,11):
    sl = 'slice_'+str(i+1)
    limit = 50*(i+1)
    ind_slice = SP500_IND_LIST[limit-50:limit]
    SP500_IND_SLICES.append(ind_slice)
    
SP500_FRAME = pd.DataFrame(SP500_DATA.loc[:,'RIC'])

SP500_FRAME['SLICE'] = 1

x = 1
k = 0
for i in SP500_FRAME.iterrows():
    SP500_FRAME.iloc[i[0],1] = 'slice_'+str(x)
    k = k+1
    if k%50 == 0:
        k = 0
        x = x+1

SP500_FRAME.to_excel('SP500_Ticketers_RIC.xlsx')
        

#%% Define Function

end = datetime.datetime.now()
start = '2000-01-01T09:00:00'

def GET_DATA_SLICE(slice_list, slice_number):
    
    slice_data = pd.DataFrame()
    for i in slice_list[slice_number-1]:
        print('Slice Stock: ' + i)
        df_stock = eikon.get_timeseries(i,start_date=start,end_date=end)
        df_stock = df_stock.loc['2010-09-08 00:00:00':,:]
        df_stock.to_sql(con=engine,name=i,if_exists='replace', index=True,schema='STOCKS')
        
        df_stock_ind = list(df_stock.columns)
        cols = []
        for j in df_stock_ind:
            cols.append(i+'_'+j)
        df_stock.columns = cols
        
        
        slice_data = pd.concat([slice_data,df_stock],axis=1)
        
        
    return slice_data
    
#%% DOWNLOAD STOCKS DATA

slice_1 = GET_DATA_SLICE(slice_list=SP500_IND_SLICES,slice_number=1)
slice_2 = GET_DATA_SLICE(slice_list=SP500_IND_SLICES,slice_number=2)
slice_3 = GET_DATA_SLICE(slice_list=SP500_IND_SLICES,slice_number=3)
slice_4 = GET_DATA_SLICE(slice_list=SP500_IND_SLICES,slice_number=4)
slice_5 = GET_DATA_SLICE(slice_list=SP500_IND_SLICES,slice_number=5)
slice_6 = GET_DATA_SLICE(slice_list=SP500_IND_SLICES,slice_number=6)
slice_7 = GET_DATA_SLICE(slice_list=SP500_IND_SLICES,slice_number=7)
slice_8 = GET_DATA_SLICE(slice_list=SP500_IND_SLICES,slice_number=8)
slice_9 = GET_DATA_SLICE(slice_list=SP500_IND_SLICES,slice_number=9)
slice_10 = GET_DATA_SLICE(slice_list=SP500_IND_SLICES,slice_number=10)
slice_11 = GET_DATA_SLICE(slice_list=SP500_IND_SLICES,slice_number=11)


#%% DOWNLOAD INDEX DATA

df_index = eikon.get_timeseries('.SP500',start_date=start,end_date=end)
df_index = df_index.loc['2010-09-08 00:00:00':,:]
df_index.to_sql(con=engine, name='.SP500', if_exists='replace', index=True, schema='MARKET_INDEX')

#%% DOWNLOAD RISK FREE DATA



fixed_income = ['US1MT=RR','US2MT=RR','US3MT=RR','US6MT=RR',
                'US1YT=RR','US2YT=RR','US3YT=RR','US5YT=RR',
                'US7YT=RR','US10YT=RR','US20YT=RR','US30YT=RR']


df_fixed = pd.DataFrame()
for i in fixed_income:
    print('Fixed_Income: ' + i)
    df = eikon.get_timeseries(i,start_date=start,end_date=end)
    df = df.loc['2010-09-08 00:00:00':,:]
    df.to_sql(con=engine,name=i,if_exists='replace', index=True,schema='FIXED_INCOME')
    
    df_ind = list(df.columns)
    cols = []
    for j in df_ind:
        cols.append(i+'_'+j)
    df.columns = cols
    df_fixed = pd.concat([df_fixed,df],axis=1)
    



#%% EXPORT TO EXCEL DATA SHEET

with pd.ExcelWriter('sabana_SP500.xlsx') as writer:  
    slice_1.to_excel(writer, sheet_name='slice_1')
    slice_2.to_excel(writer, sheet_name='slice_2')
    slice_3.to_excel(writer, sheet_name='slice_3')
    slice_4.to_excel(writer, sheet_name='slice_4')
    slice_5.to_excel(writer, sheet_name='slice_5')
    slice_6.to_excel(writer, sheet_name='slice_6')
    slice_7.to_excel(writer, sheet_name='slice_7')
    slice_8.to_excel(writer, sheet_name='slice_8')
    slice_9.to_excel(writer, sheet_name='slice_9')
    slice_10.to_excel(writer, sheet_name='slice_10')
    slice_11.to_excel(writer, sheet_name='slice_11')
    df_index.to_excel(writer, sheet_name='market_index')
    df_fixed.to_excel(writer, sheet_name='fixed_income')
    
    
