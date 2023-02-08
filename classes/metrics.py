# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 04:53:55 2023

@author: Andriu
"""

class CLASS:
    
    def __init__(self, true, pred):
        
        import pandas as pd
        import numpy as np
        from classes import autocorr
        
        self.pd = pd
        self.np = np
        self.autocorr = autocorr.CLASS
        
        self.x = true
        self.y = pred
        
        self.e = self.y - self.x
        
        self.dx = self.x - self.x.shift(1)
        self.dx.dropna(inplace=True)
        
        self.dy = self.y - self.y.shift(1)
        self.dy.dropna(inplace=True)
        
        self.mae = None
        self.da = None
        self.dis = None
        self.da_oracle = None
        self.rmse_dir = None
        self.rmse_dh = None
        self.dah = None
        
        self.df = pd.concat([self.x, self.y, self.e],axis=1)
        self.df.columns = ['NORM_SIGNAL','NORM_PREDICTION','NORM_ERROR']
        
    def get_mae(self):
        
        e = self.e
        ae = self.np.abs(e)
        mae = self.np.mean(ae)
        
        self.mae = mae
        return self.mae
    
    def get_da(self):
        
        dx = self.dx.copy()
        dy = self.dy.copy()
        
        dadf = self.pd.concat([dx,dy],axis=1)
        dadf.columns = ['dx','dy']
        dadf['compare'] = dadf['dx']*dadf['dy']
        
        def compare(x):
            if x >= 0:
                y = 1
            else:
                y = 0
            return y
        
        dadf['hit'] = dadf['compare'].apply(compare)
        
        da = self.np.mean(dadf['hit'])
        
        self.da = da
        return self.da

    def get_dis(self, tau):
        
        AC = self.autocorr(self.df, tau)    
        AC.calculate()    
        AC.get_dis()

        self.dis = AC.discrim
        return self.dis

    def get_oracle(self):
        
        dx = self.dx.copy()
        
        def compare(x):
            if x >= 0:
                y = 1
            else:
                y = 0
            return y
        
        up = dx.apply(compare)
        oracle = self.np.mean(up)
        
        self.da_oracle = oracle
        return self.da_oracle
    
    def get_binary(self):
        
        dx = self.dx.copy()
        dy = self.dy.copy()
        
        def compare(x):
            if x >= 0:
                y = 1
            else:
                y = -1
            return y
        
        bx = dx.apply(compare)
        by = dy.apply(compare)
        
        mse = self.np.mean((by - bx)**2)
        rmse = self.np.sqrt(mse)
        self.rmse = rmse
        
        return rmse
    
    def get_histeresis(self,umbral):
        
        dx = self.dx.copy()
        dy = self.dy.copy()
        
        def compare1(x):
            if x > umbral:
                y = 1
            elif x < -umbral:
                y = -1
            else:
                y = 0
            return y
        
        bx = dx.apply(compare1)
        by = dy.apply(compare1)
        
        mse_dh = self.np.mean((by-bx)**2)
        rmse_dh = self.np.sqrt(mse_dh)
        
        self.rmse_dh = rmse_dh
        
        b = bx == by
        dah = b.mean()
        self.dah = dah
        
        return self.rmse_dh, self.dah
        
        
        
        