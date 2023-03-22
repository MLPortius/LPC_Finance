# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 14:21:44 2023

@author: Andriu
"""

#%% IMPORT LIBRARIES

import os
import pandas as pd
import numpy as np
import compress_pickle as cpickle
import argparse

import statsmodels.api as sm

from sklearn.preprocessing import PolynomialFeatures

#%% SCRIPT SETUP

parser = argparse.ArgumentParser(prog= 'LPC_Determinant_Regression',
                                 description = 'Evaluates LPC Determinants with Regression',
                                 epilog = 'Created by Andriu')


parser.add_argument('-d','--dset', required=True, type=int, choices=[1,2,3])
args = parser.parse_args()

# ARGUMENTS

ds = args.dset

#%% 

infolder = 'output/determinants/'
outfolder = 'output/determinants/'

dset = 'set'+str(ds)
dpath = dset+'/'


#%% LOAD DATA

data = cpickle.load(infolder+dpath+'det_data_dict.lzma')
metrics = ['mae','da','dis']

#%% FUNCTION DEFINITION

def TO_FLOAT(x):
    y = float(x)
    return y

def SDF_TO_FDF(sdf):
    
    cols = list(sdf.columns)
    for c in cols:
        sdf[c] = sdf[c].apply(TO_FLOAT)
        
    return sdf
    

#%% NOT ROBUST LINEAR REGRESSION

for m in metrics:
    
    print('\n', m, '...')
    df = data[m]

    y = df.iloc[:,0]

    x = df.iloc[:,1:]
    x = sm.add_constant(x)

    model = sm.OLS(y, x).fit()

    pmodel = model.summary()

    print(pmodel)

    with open(outfolder + dpath + m + '/lin_reg_nr/' + 'determinants.txt', 'w') as file:
        file.write(pmodel.as_text())
        
    with open(outfolder + dpath + m + '/lin_reg_nr/' + 'determinants.latex', 'w') as file:
        file.write(pmodel.as_latex())
      
    html_table = pmodel.tables[1].as_html()
    dft = pd.read_html(html_table)[0]
    cols = list(dft.iloc[0,:])
    cols[0] = 'VARIABLE'
    dft.columns = cols
    dft = dft.iloc[1:, :]
    dft.set_index('VARIABLE', drop=True, inplace=True)
    dft = SDF_TO_FDF(dft)

    dft.to_excel(outfolder + dpath + m + '/lin_reg_nr/' + 'determinants.xlsx')


#%% NOT ROBUST POLYNOMIAL REGRESSION

for m in metrics:
    
    print('\n', m, '...')
    df = data[m]

    y = df.iloc[:,0]
    x = df.iloc[:,1:]
    
    poly = PolynomialFeatures(2)
    x2 = poly.fit_transform(x)
    x2 = pd.DataFrame(x2, index=x.index, columns = poly.get_feature_names_out())
    x2 = x2.iloc[:,1:]
    x2 = sm.add_constant(x2)
    
    model = sm.OLS(y, x2).fit()
    pmodel = model.summary()
    
    print(pmodel)

    with open(outfolder + dpath + m + '/poly_reg_nr/' + 'determinants.txt', 'w') as file:
        file.write(pmodel.as_text())
        
    with open(outfolder + dpath + m + '/poly_reg_nr/' + 'determinants.latex', 'w') as file:
        file.write(pmodel.as_latex())
      
    html_table = pmodel.tables[1].as_html()
    dft = pd.read_html(html_table)[0]
    cols = list(dft.iloc[0,:])
    cols[0] = 'VARIABLE'
    dft.columns = cols
    dft = dft.iloc[1:, :]
    dft.set_index('VARIABLE', drop=True, inplace=True)
    dft = SDF_TO_FDF(dft)
    
    dft.to_excel(outfolder + dpath + m + '/poly_reg_nr/' + 'determinants.xlsx')