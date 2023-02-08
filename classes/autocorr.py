# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 20:47:11 2023

@author: Andriu
"""


class AutoCorr:
    
    def __init__(self, data, max_tau):
        
        import pandas as pd
        import numpy as np
        import plotly as px
        
        self.pd = pd
        self.np = np
        self.px = px
        
        self.data = data
        self.max_tau = max_tau
        self.autocorres = pd.DataFrame()
        self.discrim = ()
    
    def auto_corre_serie(self, serie, max_tau):
        
        autocorres = []
        
        vector = list(serie)
        l = len(vector)
        
        for tau in range(max_tau+1):
            suma = 0
            for i in range(tau, l):
                xn = vector[i]
                xnt = vector[i-tau]
                suma += xn * xnt
            
            ac = suma / (l - tau)
            autocorres.append(ac)
        
        autocorres = self.pd.Series(autocorres)
        
        return autocorres
    
    def calculate(self):

        max_t = self.max_tau
        results = self.data
        
        autocorres = self.pd.concat([self.auto_corre_serie(results['NORM_SIGNAL'],max_t),
                                     self.auto_corre_serie(results['NORM_PREDICTION'],max_t),
                                     self.auto_corre_serie(results['NORM_ERROR'],max_t)],
                                    axis=1)
        
        autocorres.columns = ['original','predicted','error']
        
        self.autocorres = autocorres
        
    def get_dis(self):
        
        autocorr = self.autocorres
        
        dis = autocorr['error'].values[0]/autocorr['original'].values[0]
        
        self.discrim = dis
        
    def plot_original(self):
        
        fig = self. px.line(self.autocorres, title='Autocorrelaciones sin Normalizar de la Señal con intervalo %0.f'%(len(self.autocorres.index)-1))
        fig.update_xaxes(title ='tau')
        fig.update_yaxes(title ='autocorrelacion')
        plot(fig)
    
    def plot_norm(self):
        
        df = self.autocorres
        for var in list(df.columns):
            df[var] = df[var]/df[var][0]
        
        fig = self.px.line(df, title='Autocorrelaciones Normalizadas de la Señal con intervalo %0.f'%(len(df.index)-1))
        fig.update_xaxes(title ='tau')
        fig.update_yaxes(title ='autocorrelacion')
        plot(fig)