#%% IMPORTAR LIBRERIAS --------------------------------------------------------------------------------------

import os
import argparse
import pandas as pd
import numpy as np
import compress_pickle as cpickle
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

import warnings
warnings.filterwarnings('ignore')

#%% SCRIPT SETUP --------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(prog= 'LPC_Determinant_Analysis_Data_Preparation',
                                  description = 'Recopiles LPC data and creates dataframes',
                                  epilog = 'Created by Andriu')


parser.add_argument('-d','--dset', required=True, type=int, choices=[1,2,3])
parser.add_argument('-mv','--mvalues', required=True, type=str, choices=['dropna','mean','ceros'])

args = parser.parse_args()

# ARGUMENTS
ds = args.dset
mv_method = args.mvalues

#%% ENVIRONMENT SETUP ---------------------------------------------------------------------------------------

# ds = 1
# mv_method = 'dropna'

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
        
        # print(i)        
        
        temp_corr = da_temp_corr.loc[i, :]
        # print('\nCase:', temp_corr.VAR1, temp_corr.VAR2, temp_corr.CORR)

        v1c = dacorr[temp_corr.VAR1]
        v2c = dacorr[temp_corr.VAR2]
        # print('Var1:', temp_corr.VAR1, v1c)
        # print('Var2:', temp_corr.VAR2, v2c)
        
        if v1c >= v2c:
            drop = 'v2'
            # print('Dropping:', temp_corr.VAR2)
        elif v1c < v2c:
            drop = 'v1'
            # print('Dropping:', temp_corr.VAR1)

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
        
        # print(i)        
        
        temp_corr = mape_temp_corr.loc[i, :]
        # print('\nCase:', temp_corr.VAR1, temp_corr.VAR2, temp_corr.CORR)

        v1c = mapecorr[temp_corr.VAR1]
        v2c = mapecorr[temp_corr.VAR2]
        # print('Var1:', temp_corr.VAR1, v1c)
        # print('Var2:', temp_corr.VAR2, v2c)
        
        if v1c >= v2c:
            drop = 'v2'
            # print('Dropping:', temp_corr.VAR2)
        elif v1c < v2c:
            drop = 'v1'
            # print('Dropping:', temp_corr.VAR1)

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
        
        # print(i)        
        
        temp_corr = dis_temp_corr.loc[i, :]
        # print('\nCase:', temp_corr.VAR1, temp_corr.VAR2, temp_corr.CORR)

        v1c = discorr[temp_corr.VAR1]
        v2c = discorr[temp_corr.VAR2]
        # print('Var1:', temp_corr.VAR1, v1c)
        # print('Var2:', temp_corr.VAR2, v2c)
        
        if v1c >= v2c:
            drop = 'v2'
            # print('Dropping:', temp_corr.VAR2)
        elif v1c < v2c:
            drop = 'v1'
            # print('Dropping:', temp_corr.VAR1)

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

def has_indus(x):
    if 'INDUSTRY' in x:
        return 1
    else:
        return 0

def GET_ROW_TYPE(x):
    y = x.split('.')[-1]
    return y

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

indus_dict = {}

# GET INDUSTRY DUMMIES

for m in ['da','mape','dis']:
    print('\nGetting INDUS for',m)
    indus_list = []

    for i in range(len(regs[m])):

        temp_reg = regs[m][i].copy()
        temp_reg['dets']['INDUS'] = temp_reg['dets'].index.to_series().apply(has_indus)

        if temp_reg['dets']['INDUS'].sum() > 0:
            indus = 'Yes'
        else:
            indus = 'No'

        temp_reg['dets'] = temp_reg['dets'][temp_reg['dets']['INDUS'] == 0]
        temp_reg['dets'].drop(['INDUS'], axis=1, inplace=True)

        regs[m][i] = temp_reg
        indus_list.append(indus)

    indus_dict[m] = indus_list


# GET DEPENDENT VARIABLES

depvar_dict = {}

for m in ['da', 'mape', 'dis']:
    print('\nGetting DEPVAR for',m)
    depvar_list = []

    for i in range(len(regs[m])):
        depvar_list.append(m.upper())

    depvar_dict[m] = depvar_list


# GET REGRESS COLINALITY

vif_dict = {}

for m in ['da', 'mape', 'dis']:
    
    print('\nGetting VIF for',m)

    vif_list = []
    for i in range(len(dataframes[m])):

        X = dataframes[m][i].copy()

        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [vif(X.values, i) for i in range(len(X.columns))]
        
        vif_mean = np.round(vif_data.VIF.mean(),2)
        vif_nv = vif_data[vif_data['VIF'] > 5].feature.count() / vif_data.feature.count()
        vif_am = vif_data[vif_data.VIF > vif_mean].feature.count() / vif_data.feature.count()
        vif_max = np.round(vif_data.VIF.max(),2)

        vif_serie = pd.Series([vif_mean, vif_max, vif_nv, vif_am], index=['VIF_MEAN', 'VIF_MAX', 'VIF_N_VAR_ABOVE_5', 'VIF_N_VAR_ABOVE_MEAN'])

        vif_list.append(vif_serie)

    vif_dict[m] = vif_list


# GET DET DATAFRAMES

df1_dict = {}

for m in ['da','mape','dis']:

    print('\nGetting DET DF for',m)
    
    df1_list = []

    for i in range(len(regs[m])):

        dft = regs[m][i]['dets'].copy()
        df1_list.append(dft)

    df1_dict[m] = df1_list

# GET REG DATAFRAMES

df0_dict = {}

for m in ['da','mape','dis']:
    print('\nGetting METRICS DF for',m)
    
    df0_list = []
    for i in range(len(regs[m])):
        dft = regs[m][i]['metrics'].copy()
        df0_list.append(dft)

    df0_dict[m] = df0_list

#%% SORT VARS DATAFRAMES

df1v_dict = {}
df1c_dict = {}

for m in ['da','mape','dis']:
    print('\nSorting DETS DF for',m)
    
    df1v_list = []
    df1c_list = []

    for i in range(len(df1_dict[m])):

        dft = df1_dict[m][i].copy()

        dft1 = dft.COEF.to_frame()
        dft1.index = [x+'.[1COEF]' for x in list(dft1.index)]
        dft1.columns = ['REG']

        dft2 = dft.STD.to_frame()
        dft2.index = [x+'.[2STD]' for x in list(dft2.index)]
        dft2.columns = ['REG']

        dft3 = dft.P_VALUE.to_frame()
        dft3.index = [x+'.[3PVALUE]' for x in list(dft3.index)]
        dft3.columns = ['REG']

        dftt = pd.concat([dft1, dft2, dft3], axis=0)
        dftt.sort_index(ascending=True, inplace=True)

        dfttc = dftt.loc[[x for x in list(dftt.index) if (len(x.split('const')) > 1)]]
        dfttv = dftt.loc[[x for x in list(dftt.index) if (len(x.split('const')) <= 1)]]

        df1v_list.append(dfttv)
        df1c_list.append(dfttc)

    df1v_dict[m] = df1v_list
    df1c_dict[m] = df1c_list

#%% AGREGATE METRICS DFS

tables_dict = {}

for m in ['da','mape','dis']:
    print('\nGetting tables for',m)

    # METRICS TABLE
    dft = df0_dict[m][0].copy()
    for d in df0_dict[m][1:]:
        dft = pd.concat([dft, d.copy()], axis=1)
    df_metrics = dft.copy()

    # CONST TABLE
    dft = df1c_dict[m][0].copy()
    for d in df1c_dict[m][1:]:
        dft = pd.concat([dft, d.copy()], axis=1)
    df_const = dft.copy()
    df_const = df_const.astype(float)
    
    # VARS TABLE
    dft = df1v_dict[m][0].copy()
    for d in df1v_dict[m][1:]:
        dft = pd.concat([dft, d.copy()], axis=1)
    df_vars = dft.copy()

    dft = pd.DataFrame(df_vars.index, columns=['VARREG'])
    dft['VAR'] = [x.split('.')[0] for x in list(dft.VARREG)]
    dft.set_index('VAR', inplace=True)

    dftt = pd.merge(dft,info.loc[:,['NAME', 'TYPE', 'SUBTYPE']], left_index=True, right_index=True)
    dftt.set_index('VARREG', inplace=True, drop=False)

    dftt['ROW_TYPE'] = dftt.VARREG.apply(GET_ROW_TYPE) 

    dftt['VAR'] = dftt.VARREG.apply(lambda x: x.split('.')[0])
    dftt['VAR_ID'] = dftt.VAR + '.' + '.' + dftt.TYPE + '.' + dftt.SUBTYPE + '.' + dftt.ROW_TYPE
    dftt = dftt.loc[:,'VAR_ID']

    df_vars = pd.concat([df_vars, dftt], axis=1)
    df_vars.set_index('VAR_ID', inplace=True, drop=True)

    df_vars = df_vars.astype(float)

    # VIF TABLE
    dft = vif_dict[m][0].copy()
    for d in vif_dict[m][1:]:
        dft = pd.concat([dft, d.copy()], axis=1)
    df_vif = dft.copy()

    # INDUS TABLE
    df_indus = pd.DataFrame(indus_dict[m]).T
    df_indus.index = ['INDUSTRIES']

    # DEPVAR TABLE
    df_depvar = pd.DataFrame(depvar_dict[m]).T
    df_depvar.index = ['DEPVAR']
    df_depvar

    # AGGREGATE TABLES
    cols = ['REG_'+str(i+1) for i in range(len(df_depvar.columns))]

    df_depvar.columns = cols
    df_metrics.columns = cols
    df_const.columns = cols
    df_vars.columns = cols
    df_indus.columns = cols
    df_vif.columns = cols

    empty_df = pd.DataFrame([np.nan for i in range(len(df_depvar.columns))], index=cols).T
    empty_df.index = ['']
    empty_df

    dfr = pd.concat([df_depvar, empty_df,
                    df_metrics, empty_df,
                    df_const, df_vars, empty_df,
                    df_indus, empty_df,
                    df_vif], axis=0)
    
    dfr.reset_index(inplace=True)
    cols = list(dfr.columns)
    cols[0] = 'TABLE'
    dfr.columns = cols
    
    dfr.TABLE = [x.upper() for x in list(dfr.TABLE)]

    tables_dict[m] = dfr.copy()

#%% EXPORT TABLES
for m in ['da','mape','dis']:
    tables_dict[m].to_excel('output/determinants/'+dpath+m+'/'+dset+'_'+m+'_table.xlsx',index=False)
