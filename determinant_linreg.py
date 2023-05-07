#%% IMPORTAR LIBRERIAS --------------------------------------------------------------------------------------

import os
import argparse
import pandas as pd
import numpy as np
import compress_pickle as cpickle
import statsmodels.api as sm

#%% SCRIPT SETUP --------------------------------------------------------------------------------------------

# parser = argparse.ArgumentParser(prog= 'LPC_Determinant_Analysis_Data_Preparation',
#                                   description = 'Recopiles LPC data and creates dataframes',
#                                   epilog = 'Created by Andriu')


# parser.add_argument('-d','--dset', required=True, type=int, choices=[1,2,3])
# parser.add_argument('-mv','--mvalues', required=True, type=str, choices=['dropna','mean','ceros'])

# args = parser.parse_args()

# ARGUMENTS
# ds = args.dset
# mv_method = args.mvalues

#%% ENVIRONMENT SETUP ---------------------------------------------------------------------------------------

ds = 1
mv_method = 'dropna'

dset = 'set'+str(ds)
dpath = dset+'/' 
fpath = 'output/determinants/' + dpath

np.random.seed(14)

#%% CARGAR DATOS --------------------------------------------------------------------------------------------

with open(fpath+'det_main_data.lzma', 'rb') as f:
    data = cpickle.load(f)

if mv_method == 'dropna':
    data.dropna(axis=0, inplace=True)
elif mv_method == 'mean':
    data.fillna(data.mean(), inplace=True)
elif mv_method == 'ceros':
    data.fillna(0, inplace=True)

info = pd.read_excel('output/determinants/variables_structure.xlsx')
info.set_index('VARIABLE', inplace=True)

#%% GET TYPES AND SUBTYPES ----------------------------------------------------------------------------------

info_ind = info[info.REGRESSION == 'indvar']
info_dep = info[info.REGRESSION == 'depvar']

indvars = list(info_ind.index)
indv = {}

for i in indvars:
    indv[i] = [info_ind.loc[i,'TYPE'], info_ind.loc[i,'SUBTYPE']]

types = list(info_ind.TYPE.unique())
subtypes = list(info_ind.SUBTYPE.unique())

indvars = indv

del(indv, i, f)

#%%  GET VARIABLES OF TYPES AND SUBTYPES --------------------------------------------------------------------

variables = {}

for t in types:

    sdict = {}
    for s in subtypes:
        _ = info[info.TYPE == t]
        sdict[s] = list(_[_.SUBTYPE == s].index)

    variables[t] = sdict

del(t, s, _, sdict)

#%% GET 4 VARIABLES SETS ------------------------------------------------------------------------------------

df = pd.DataFrame(variables)

def random_element(lst):
    return lst[np.random.randint(len(lst))]

# 1 regresion con todo
selected_0 = list(info_ind.index)

# 1 regresion por tipo

type_vars = {}

for t in types:
    type_vars[t] = list(info[info.TYPE == t].index)

selected_1 = []
for t in types:
    _ = random_element(type_vars[t])
    selected_1.append(_)

selected_2 = []
for t in types:
    _ = random_element(type_vars[t])
    
    if _ in selected_1:
        _ = random_element(type_vars[t])
    
    selected_2.append(_)

# 1 regresion por subtipo

sub_vars = {}
for s in subtypes:
    sub_vars[s] = list(info[info.SUBTYPE == s].index)

selected_3 = []
for s in subtypes:
    _ = random_element(sub_vars[s])
    selected_3.append(_)

selected_4 = []
for s in subtypes:
    _ = random_element(sub_vars[s])
    
    if _ in selected_3:
        _ = random_element(sub_vars[s])
    
    selected_4.append(_)


#%% DATA PREPARATION ---------------------------------------------------------------------------------------

data_dep = data.loc[:, list(info_dep.index)]
data_dep

da = data_dep.OPT_DA
mape = data_dep.OPT_MAPE
dis = data_dep.OPT_DIS

industries = list(data.columns[data.columns.str.contains('INDUSTRY')])
industriesdf = data.loc[:,industries]

data_ind = data[info[info.REGRESSION == 'indvar'].index]

#%% SELECTED DATAFRAMES ------------------------------------------------------------------------------------

# 1 sin los bajamente correlacionados con la variable dep

percent = int(np.round(0.5 * len(data_ind.columns), 0))

dt = pd.concat([da, data_ind], axis=1)
selected_6_da = list(dt.corr().iloc[0, 1:].abs().sort_values(ascending=False)[:percent].index)

dt = pd.concat([mape, data_ind], axis=1)
selected_6_mape = list(dt.corr().iloc[0, 1:].abs().sort_values(ascending=False)[:percent].index)

dt = pd.concat([dis, data_ind], axis=1)
selected_6_dis = list(dt.corr().iloc[0, 1:].abs().sort_values(ascending=False)[:percent].index)

# 1 sin los altamente correlacionados entre ellos

dt = data_ind.copy()
corr = dt.corr()

vars = []

for i in list(corr.index):

    for j in list(corr.columns):

        listo = [i, j]
        listo.sort()
        listo.append(np.abs(corr.loc[i, j]))

        vars.append(listo)

corr = pd.DataFrame(vars)
corr.columns = ['VAR1', 'VAR2', 'CORR']
corr = corr[corr.CORR != 1]

corr = corr.drop_duplicates()

corr.sort_values('CORR', ascending=False, inplace=True)
corr = corr[corr.CORR >= 0.2]

dtda = pd.concat([da, data_ind], axis=1)
dtmape = pd.concat([mape, data_ind], axis=1)
dtdis = pd.concat([dis, data_ind], axis=1)

dacorr = dtda.corr().iloc[0, 1:].abs()
mapecorr = dtmape.corr().iloc[0, 1:].abs()
discorr = dtdis.corr().iloc[0, 1:].abs()

# DA VARIABLES

da_temp_corr = corr.copy()
da_temp_corr.index = ['ID_'+str(x) for x in list(da_temp_corr.index)]

drops = []

for i in list(da_temp_corr.index):

    if i in list(da_temp_corr.index):
        
        print(i)        
        
        temp_corr = da_temp_corr.loc[i, :]
        print('\nCase:', temp_corr.VAR1, temp_corr.VAR2, temp_corr.CORR)

        v1c = dacorr[temp_corr.VAR1]
        v2c = dacorr[temp_corr.VAR2]
        print('Var1:', temp_corr.VAR1, v1c)
        print('Var2:', temp_corr.VAR2, v2c)
        
        if v1c >= v2c:
            drop = 'v2'
            print('Dropping:', temp_corr.VAR2)
        elif v1c < v2c:
            drop = 'v1'
            print('Dropping:', temp_corr.VAR1)

        if drop == 'v1':
            drops.append(temp_corr.VAR1)
            da_temp_corr = da_temp_corr[da_temp_corr.VAR1 != temp_corr.VAR1] 
            da_temp_corr = da_temp_corr[da_temp_corr.VAR2 != temp_corr.VAR1] 

        elif drop == 'v2':
            drops.append(temp_corr.VAR2)
            da_temp_corr = da_temp_corr[da_temp_corr.VAR2 != temp_corr.VAR2]
            da_temp_corr = da_temp_corr[da_temp_corr.VAR1 != temp_corr.VAR2]

selected_7_da = list(selected_0)
for d in drops:
    selected_7_da.pop(selected_7_da.index(d))


# MAPE VARIABLES

mape_temp_corr = corr.copy()
mape_temp_corr.index = ['ID_'+str(x) for x in list(mape_temp_corr.index)]

drops = []

for i in list(mape_temp_corr.index):

    if i in list(mape_temp_corr.index):
        
        print(i)        
        
        temp_corr = mape_temp_corr.loc[i, :]
        print('\nCase:', temp_corr.VAR1, temp_corr.VAR2, temp_corr.CORR)

        v1c = mapecorr[temp_corr.VAR1]
        v2c = mapecorr[temp_corr.VAR2]
        print('Var1:', temp_corr.VAR1, v1c)
        print('Var2:', temp_corr.VAR2, v2c)
        
        if v1c >= v2c:
            drop = 'v2'
            print('Dropping:', temp_corr.VAR2)
        elif v1c < v2c:
            drop = 'v1'
            print('Dropping:', temp_corr.VAR1)

        if drop == 'v1':
            drops.append(temp_corr.VAR1)
            mape_temp_corr = mape_temp_corr[mape_temp_corr.VAR1 != temp_corr.VAR1] 
            mape_temp_corr = mape_temp_corr[mape_temp_corr.VAR2 != temp_corr.VAR1] 

        elif drop == 'v2':
            drops.append(temp_corr.VAR2)
            mape_temp_corr = mape_temp_corr[mape_temp_corr.VAR2 != temp_corr.VAR2]
            mape_temp_corr = mape_temp_corr[mape_temp_corr.VAR1 != temp_corr.VAR2]

selected_7_mape = list(selected_0)
for d in drops:
    selected_7_mape.pop(selected_7_mape.index(d))


# DIS VARIABLES

dis_temp_corr = corr.copy()
dis_temp_corr.index = ['ID_'+str(x) for x in list(dis_temp_corr.index)]

drops = []

for i in list(dis_temp_corr.index):

    if i in list(dis_temp_corr.index):
        
        print(i)        
        
        temp_corr = dis_temp_corr.loc[i, :]
        print('\nCase:', temp_corr.VAR1, temp_corr.VAR2, temp_corr.CORR)

        v1c = discorr[temp_corr.VAR1]
        v2c = discorr[temp_corr.VAR2]
        print('Var1:', temp_corr.VAR1, v1c)
        print('Var2:', temp_corr.VAR2, v2c)
        
        if v1c >= v2c:
            drop = 'v2'
            print('Dropping:', temp_corr.VAR2)
        elif v1c < v2c:
            drop = 'v1'
            print('Dropping:', temp_corr.VAR1)

        if drop == 'v1':
            drops.append(temp_corr.VAR1)
            dis_temp_corr = dis_temp_corr[dis_temp_corr.VAR1 != temp_corr.VAR1] 
            dis_temp_corr = dis_temp_corr[dis_temp_corr.VAR2 != temp_corr.VAR1] 

        elif drop == 'v2':
            drops.append(temp_corr.VAR2)
            dis_temp_corr = dis_temp_corr[dis_temp_corr.VAR2 != temp_corr.VAR2]
            dis_temp_corr = dis_temp_corr[dis_temp_corr.VAR1 != temp_corr.VAR2]

selected_7_dis = list(selected_0)
for d in drops:
    selected_7_dis.pop(selected_7_dis.index(d))

#%% DATAFRAME SELECTIONS ---------------------------------------------------------

da_selections = [selected_0, selected_1, selected_2, selected_3, 
                 selected_4, selected_6_da, selected_7_da]

mape_selections = [selected_0, selected_1, selected_2, selected_3,
                   selected_4, selected_6_mape, selected_7_mape]

dis_selections = [selected_0, selected_1, selected_2, selected_3,
                  selected_4, selected_6_dis, selected_7_dis]


da_selections_industry = []
for s in da_selections:
    da_selections_industry.append(s)
    da_selections_industry.append(s + industries)

mape_selections_industry = []
for s in mape_selections:
    mape_selections_industry.append(s)
    mape_selections_industry.append(s + industries)

dis_selections_industry = []
for s in dis_selections:
    dis_selections_industry.append(s)
    dis_selections_industry.append(s + industries)


da_dataframes = []
for s in da_selections_industry:
    tempdf = data.loc[:, s]
    tempdf = pd.concat([tempdf, da], axis=1)
    da_dataframes.append(tempdf)

mape_dataframes = []
for s in mape_selections_industry:
    tempdf = data.loc[:, s]
    tempdf = pd.concat([tempdf, mape], axis=1)
    mape_dataframes.append(tempdf)

dis_dataframes = []
for s in dis_selections_industry:
    tempdf = data.loc[:, s]
    tempdf = pd.concat([tempdf, dis], axis=1)
    dis_dataframes.append(tempdf)


del(corr, d, da, da_selections, da_selections_industry, mape, mape_selections, mape_selections_industry, dis, dis_selections, dis_selections_industry,
    da_temp_corr, dacorr, v1c, v2c, tempdf, selected_0, selected_1, selected_2, selected_3, selected_4, selected_6_da, selected_7_da,
    selected_6_mape, selected_7_mape, selected_6_dis, selected_7_dis, df, dis_temp_corr, discorr, drop, drops, dt, dtda, dtdis, dtmape, i, 
    industries, industriesdf, info_dep, info_ind, j, listo, mape_temp_corr, mapecorr, percent, s, sub_vars, subtypes, t, temp_corr, 
    type_vars, types, variables, vars)


dataframes = {}
dataframes['da'] = da_dataframes
dataframes['mape'] = mape_dataframes
dataframes['dis'] = dis_dataframes

del(da_dataframes, mape_dataframes, dis_dataframes)

#%% FUNCITON DEFINITION ---------------------------------------------------------
def RUN_LINREG(dataframe):

    X = dataframe.iloc[:, :-1]
    X = sm.add_constant(X)

    Y = dataframe.iloc[:, -1]
    
    model = sm.OLS(Y, X).fit()
    pmodel = model.summary()

    return pmodel

def REGTABLES_TO_DF(reg):

    html0 = reg.tables[0].as_html()
    html1 = reg.tables[1].as_html()

    df1 = pd.read_html(html1)[0]
    df1 = df1.iloc[:, [0, 1, 2, 4]]
    df1.columns = ['VARIABLE', 'COEF', 'STD', 'P_VALUE']

    df1 = df1.iloc[1:, :]
    df1.set_index('VARIABLE', drop=True, inplace=True)

    df0 = pd.read_html(html0)[0]
    df0_1 = df0.iloc[:, [0, 1]]
    df0_2 = df0.iloc[:, [2, 3]]
    df0_1.columns = ['METRIC','VALUE']
    df0_2.columns = ['METRIC','VALUE']
    df0 = pd.concat([df0_1, df0_2], axis=0)

    df0 = df0.iloc[[5, 9, 10],:]
    df0.set_index('METRIC', drop=True, inplace=True)
    df0.index = ['Observations', 'R2', 'Adjusted R2']

    dicto = {'metrics':df0, 'dets':df1}

    return dicto

#%% RUN LINREG -------------------------------------------------------------------

da_regs = []

for df in dataframes['da']:
    pmodel = RUN_LINREG(df)
    tables = REGTABLES_TO_DF(pmodel)
    da_regs.append(tables)

mape_regs = []

for df in dataframes['mape']:
    pmodel = RUN_LINREG(df)
    tables = REGTABLES_TO_DF(pmodel)
    mape_regs.append(tables)

dis_regs = []

for df in dataframes['dis']:
    pmodel = RUN_LINREG(df)
    tables = REGTABLES_TO_DF(pmodel)
    dis_regs.append(tables)

regs = {}
regs['da'] = da_regs
regs['mape'] = mape_regs
regs['dis'] = dis_regs

del(da_regs, mape_regs, dis_regs, df, pmodel, tables)

#%% CREATE TABLES FROM DATAFRAMES

metric = 'da'
reg = 1

# GET INDUSTRY DUMMIES

temp_reg = regs[metric][reg]
temp_reg

incv = list(temp_reg['dets'].index)

if len(incv[-1].split('INDUSTRY')) > 1:
    indus = True

if indus = True:
    regs[metric][reg]['dets']




















#%%

#%%
regs['da'][1]['dets']

#%%

df0s = []
df1s = []
dvars = []
indus = []
dsets = []

for i in range(len(regs['da'])):
               
    mod = regs['da'][i]['dets']

    coefs = mod.COEF.to_frame()
    coefs.index = [x+'_[1COEF]' for x in list(coefs.index)]
    coefs.columns = ['METRIC']

    stds = mod.STD.to_frame()
    stds.index = [x+'_[2STD]' for x in list(stds.index)]
    stds.columns = ['METRIC']

    ps = mod.P_VALUE.to_frame()
    ps.index = [x+'_[3PVALUE]' for x in list(ps.index)]
    ps.columns = ['METRIC']

    mod2 = pd.concat([coefs, stds, ps], axis=0)

    df0s.append(mod2)

    nod = regs['da'][i]['metrics']
    df1s.append(nod)

    dvars.append('DA')
    dsets.append(dset)

    if len(coefs.index) > 17:
        indus.append('Yes')
    else:
        indus.append('No')

df0_da = df0s[0]
for d in df0s[1:]:
    df0_da = pd.concat([df0_da, d], axis=1)

df1_da = df1s[0]
for d in df1s[1:]:
    df1_da = pd.concat([df1_da, d], axis=1)

dfvars = pd.DataFrame(dvars).T
dfindus = pd.DataFrame(indus).T
dfsets = pd.DataFrame(dsets).T



#%%

df0_da.columns = ['COL_'+str(x) for x in range(len(df0_da.columns))]
df0_da.reset_index(inplace=True, drop=False)

df1_da.columns = ['COL_'+str(x) for x in range(len(df1_da.columns))]
dfvars.columns = ['COL_'+str(x) for x in range(len(dfvars.columns))]
dfindus.columns = ['COL_'+str(x) for x in range(len(dfindus.columns))]
dfsets.columns = ['COL_'+str(x) for x in range(len(dfsets.columns))]

df_da = pd.concat([dfvars, df0_da, df1_da, dfsets, dfindus], axis=0, ignore_index=True)

cols = list(df_da.columns)
cols[-1] = 'VARIABLE'
df_da.columns = cols

#%%

def has_ind(x):
    if type(x) == str:
        if 'INDUSTRY' in x:
            return True
        else:
            return False
    return False

df_da['DROP'] = df_da.VARIABLE.apply(has_ind)

df_da = df_da[df_da.DROP == False]
df_da.drop(['DROP'], axis=1, inplace=True)

#%%
df_da.VARIABLE.fillna('VARIABLE', inplace=True)

#%%

df_da.VARIABLE
#%%
# List with all columns in df dataframe that have 'INDUSTRY' in it
list(df_da.columns[df_da.columns.str.contains('INDUSTRY')])




#%%

i = 0
nod = regs['da'][i]['metrics']
