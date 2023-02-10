# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 20:50:24 2023

@author: Andriu
"""

class CLASS:
    
    def __init__(self, serie):
        
        # LIBRERIAS
        import pandas as pd
        import numpy as np
        self.pd = pd
        self.np = np
        
        # INICIALIZACION
        self.serie = serie/1000000

        # RESULTADOS
        self.vsum = None
        self.vstd = None
        self.vmean = None
        self.vcv = None
        
    def get_metrics(self):
        
        self.vsum = self.np.sum(self.serie)
        self.vstd = self.np.std(self.serie)
        self.vmean = self.np.mean(self.serie)
        self.vcv = self.vstd/self.mean

        vmetrics = [self.vsum, self.vstd, self.mean, self.vcv]
        
        return vmetrics