# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 20:52:11 2023

@author: Andriu
"""


class CLASS:
    
    def __init__(self, signal, menos, dmas, dmen):
        
        import pandas as pd
        import numpy as np
        
        self.np = np
        self.pd = pd
        
        self.signal = signal
        self.menos = menos
        self.dmas = dmas
        self.dmen = dmen
        self.mas = None
        
    def get_corrcoefs(self):
        
        # Calculo X[n+1] - X[n]
        mas = self.np.append([0],self.np.diff(self.signal))
        mas_serie = self.pd.Series(mas)
        mas_serie.index = self.signal.index
        self.mas = mas_serie
        
        # Correlacion entre Derivada en N y Diferencia X[n] - X[n-1]
        cc_dmen_menos = self.np.corrcoef(self.dmen, self.menos)[0][1]
    
        # Correlacion entre Derivada en N y Diferencia X[n+1] - X[n]
        cc_dmen_mas = self.np.corrcoef(self.dmen, self.mas)[0][1]
        
        # Correlacion entre Diferencia X[n] - X[n-1] y Diferencia X[n+1] - X[n]
        cc_menos_mas = self.np.corrcoef(self.menos, self.mas)[0][1]
        
        # Correlacion entre Derivada en N+1 y Derivada en N
        cc_dmas_dmen = self.np.corrcoef(self.dmas, self.dmen)[0][1]
        
        # Correlacion entre Derivada en N+1 y Diferencia X[n+1]-X[n]
        cc_dmas_mas = self.np.corrcoef(self.dmas, self.mas)[0][1]
        
        results = [cc_dmen_menos, cc_dmen_mas, cc_menos_mas, 
                   cc_dmas_dmen, cc_dmas_mas]
        
        return results