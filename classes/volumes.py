# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 20:50:24 2023

@author: Andriu
"""

vols = pd.read_excel(data_folder+'SP500_VOLS_FILTERED.xlsx')
vols.set_index('Date',inplace=True,drop=True)
vols = vols/1000000

vsum = vols.sum(axis=0)
vstd = vols.std(axis=0)
vmean = vols.mean(axis=0)
vcv = vstd/vmean

vdf = pd.concat([vsum,vmean,vstd,vcv],axis=1)
vdf.columns = ['VOL_SUM','VOL_MEAN','VOL_STD','VOL_CV']