# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 20:47:20 2023

@author: Andriu
"""

class LPCgrid:
    
    def __init__(self, data, vols, label, ws_list, p_list, norm=True):
        
        
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
        
        # INICIALIZACION
        self.label = label
        self.serie = data.dropna(axis=0)
        self.lenght = len(self.serie)
        self.norm = norm
        
        self.vols = vols.dropna(axis=0)

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
        self.run_time = np.nan
        
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
            gsum = pd.concat([gsum,m.summary()],axis=0)
        return gsum

    def add_volumes(self):
        
        