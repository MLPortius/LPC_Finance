# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 20:47:20 2023

@author: Andriu
"""

class CLASS:
    
    def __init__(self, dataset, label, ws_list, p_list, norm=True):
        
        # LIBRERIAS
        import numpy as np
        import pandas as pd
        import time
        self.np = np
        self.pd = pd
        self.time = time
        
        # MODULOS
        from classes import rolling
        self.rolling = rolling.CLASS
        
        from classes import volumes
        self.volumes = volumes.CLASS
        
        # INICIALIZACION
        self.label = label
        self.serie = dataset['close'].loc[:,label]
        self.vols = dataset['vols'].loc[:,label]
        
        self.lenght = len(self.serie)
        self.norm = norm
        
        self.serie.dropna(axis=0, inplace=True)
        self.vols.dropna(axis=0, inplace=True)

        # AJUSTE DE GRILLA        
        grid = []
        
        for ws in ws_list:
            if ws<=self.lenght:
                for p in p_list:
                    if ws >= (p+30):
                        grid.append((ws,p))
                        
        self.grid = grid
        
        # CONTROL
        self.best = None
        self.roll_list = []
        self.gsum = None
        
    def grid_search(self,ntype):
        
        models = []
        
        for tupla in self.grid:
            
            print("Stage: "+self.label+" "+str(tupla[0])+" "+str(tupla[1]))
            
            start = self.time.time()
            LPC = self.rolling(self.serie,label=self.label+"_"+str(tupla[0])+"_"+str(tupla[1]))
            LPC.create(tupla[0])
            
            if self.norm:
                LPC.normalize()
            
            LPC.fit(tupla[1])
            LPC.get_coefs()
            LPC.predict()
            LPC.get_metrics()
            #LPC.calc_desv_rs()
            
            end = self.time.time()
            elapsed = self.np.round(end-start,2)
            
            print("Time elapsed: ",elapsed)    

            LPC.clean()
            
            models.append(LPC)
        
        self.roll_list = models
    
    def global_summary(self):        
        gsum = self.roll_list[0].summary()
        for m in self.roll_list[1:]:
            gsum = self.pd.concat([gsum,m.summary()],axis=0)
        self.gsum = gsum

    def add_volumes(self):
        
        gsum = self.gsum.copy()
        N = len(gsum.index)
        
        v = self.volumes(self.vols)
        vmetrics = v.get_metrics()
        
        vsum = self.pd.Series([vmetrics[0]] * N)
        vstd = self.pd.Series([vmetrics[1]] * N)
        vmean = self.pd.Series([vmetrics[2]] * N)
        vcv = self.pd.Series([vmetrics[3]] * N)
        
        vdf = self.pd.concat([vsum,vstd,vmean,vcv],axis=1)
        vdf.columns = ['vsum','vstd','vmean','vcv']
        vdf.index = gsum.index
        
        gsum = self.pd.concat([gsum,vdf],axis=1)
        
        self.gsum = gsum
        
        return gsum
    
    def add_histograms(self):
        
        print('xd...')