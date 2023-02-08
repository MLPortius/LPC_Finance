# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 20:44:50 2023

@author: Andriu
"""

#%% VENTANA CLASS

class CLASS:
    
    def __init__(self,data):
        
        import numpy as np
        import pandas as pd
        from scipy import linalg
        
        self.np = np
        self.pd = pd
        self.linalg = linalg
        
        self.serie = data
        self.vector = np.asarray(data)
        self.coefs = np.asarray([])
        self.pred = None
        self.index = list(data.index)
        self.p = None
        self.mean = None
        self.std = None
        self.norm = False
        self.r_vector = None
        self.r_vector2 = None
        self.mean_rs = None
        self.predicted = None
        self.menos = None
        self.dmas = None
        self.dmen = None
        
    def normalize(self):
        data = self.vector
        media = self.np.mean(data)
        desv = self.np.std(data)

        self.mean = media
        self.std = desv

        self.serie = (self.serie - media)/desv
        self.vector = (self.vector - media)/desv

        self.norm = True

    def fit_coef(self,p):
        
        window = self.vector
        N = len(window)
        
        r = []
        r2 = []
        for i in range(0,p+1):
            r_i = 0
            for n in range(i,N):
                r_i += window[n]*window[n-i]

            r.append(r_i)
            r2.append((r_i)/(len(window)-i))
        
        r = self.np.asarray(r)
        
        r_vector_target = r[1:]
        r_vector_feature = r[:p]
        
        r_matrix = self.linalg.toeplitz(r_vector_feature)
        
        w = self.np.linalg.inv(r_matrix).dot(r_vector_target)
        a = -w 
        
        a = self.np.insert(a,0,1)
        
        coef_lpc = a.T
        coefs = list(coef_lpc)
        
        for i in range(len(coefs)-1):
            coefs[i+1] = coefs[i+1] * -1
            
        self.coefs = coefs
        self.p = p
        self.r_vector = r
        self.r_vector2 = r2
        self.mean_rs = self.np.mean(r2)
        
    def apply_coefs(self, serie, coefs):
        
        serie = list(serie)
        coefs = list(coefs)
        
        p = self.p

        pred = 0
        for i in range(p): 
            pred += serie[p-i-1]*coefs[i+1]
            
        return pred

    def predict(self):
        
        p = self.p
        coefs = self.coefs
        
        # Vector de datos
        data = list(self.vector)        
        l = len(data)

        predicts = []
        for i in range(l-p+1):
            d = data[i:i+p]
            pred = self.apply_coefs(d,coefs)
            predicts.append(pred)
            del(i,d,pred)
            
        preds = predicts[:-1]
        last = predicts[-1]
        
        self.predicted = preds

        # Prediccion final
        if self.norm:
            unscaled_pred = last*self.std + self.mean
            self.pred = unscaled_pred
        else:
            self.pred = last
    
        # Calculo de la derivada
        self.menos = data[-1] - data[-2]
        self.dmas = predicts[-1] - predicts[-2]
        self.dmen = predicts[-2] - predicts[-3]
        
    def clean(self):
        self.serie = 0
        self.vector = 0