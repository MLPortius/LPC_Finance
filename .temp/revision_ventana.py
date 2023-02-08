# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 22:36:36 2023

@author: Andriu
"""

#%% class

from classes import metrics
import pandas as pd

df = pd.read_excel('.temp/revision.xlsx',sheet_name='serie m')

met = metrics.CLASS
m = met(true=df['x'], pred=df['y'])

mae = m.get_mae()
da = m.get_da()
dis = m.get_dis(5)
dao = m.get_oracle()
rmse_dir = m.get_binary()
rmse_dh, dah = m.get_histeresis(0.05)

#%%

import pandas as pd

s = pd.Series([1,2,3,4,5,6,7,8,9],index=['ind1','ind2','ind3','ind4','ind5',
                                         'ind6','ind7','ind8','ind9'])

s1 = pd.Series([0,0,0,0,0],index=['ind5','ind6','ind7','ind8','ind9'])
s2 = pd.Series([0,0,0,0,0],index=['ind5','ind6','ind7','ind8','ind9'])

df = pd.DataFrame([s, s1, s2]).T
df.columns = ['VAR1','VAR2','VAR3']
df.dropna(inplace=True)
