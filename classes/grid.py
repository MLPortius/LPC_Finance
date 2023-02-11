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
        
        from classes import histograms
        self.histograms = histograms.CLASS
        
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
        self.vdf = None
        self.hdf = None
        self.hs = None
        
    def grid_search(self):
        
        models = []
        
        for tupla in self.grid:
            
            print("Stage: "+self.label+" "+str(tupla[0])+" "+str(tupla[1]))
            
            start = self.time.time()
            LPC = self.rolling(self.serie,label=self.label)
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
            
        gsum._set_axis_name('stock')
        
        self.gsum = gsum

    def add_volumes(self):
        
        v = self.volumes(self.vols)
        vmetrics = v.get_metrics()
        
        vdf = self.pd.DataFrame(vmetrics).T
        vdf.columns = ['vsum','vstd','vmean','vcv']
        vdf.index = [self.label]
        vdf._set_axis_name('stock',inplace=True)
        
        self.vdf = vdf
        
    def add_histograms(self):
        
        h = self.histograms(self.serie)
        h.get_values()
        h.get_histograms()
        
        self.hdf = h.hmetrics
        self.hdf.index = [self.label]
        self.hdf._set_axis_name('stock',inplace=True)
        
        self.hs = h.histograms