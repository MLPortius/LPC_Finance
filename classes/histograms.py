# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 13:36:08 2023

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
        self.serie = serie
        
        # CONTROL
        self.values = None
        self.histograms = None
        self.hmetrics = None
        
    def get_values(self):
        # Esta funcion cuenta las rachas, y las guarda según sean positivos o negativos
        
        dif = self.serie - self.serie.shift(1)
        dif.dropna(axis=0, inplace=True)
        
        deltas = list(dif)      # Los guarda en formato lista
        
        positives = []          # Aqui se guardaran las rachas positivas
        negatives = []           # Aqui se guardaran las rachas negativas 
        totales = []            # Aqui se guardaran todas las rachas 
        
        cs = 1 # Se comienza con el minimo (1 periodo de duracion)
        
        for i in range(len(deltas)-1):
            
            d0 = deltas[i]      # Este cambio
            d1 = deltas[i+1]    # El cambio siguiente
            
            change = d0*d1      # Se compara el signo
        
            # Si change es positivo, actual y siguiente tienen el mismo signo (racha)
            
            if change >= 0:
                cs += 1         # Si poseen el mismo signo, este cambio aumenta la racha
            
            # Si la siguiente cortara la racha actual, se terminó, se deben guardar los actuales valores y reiniciar los contadores
            
            else:
                totales.append(cs) # Se guarda la duracion de racha actual
                
                if d0 >= 0:
                    positives.append(cs) # Si el ultimo valor de la racha era positivo, se guarda la duracion en la lista de rachas positivas
                else:
                    negatives.append(cs) # Si el ultimo valor de la racha era negativo, se guarda la duracion en la lista de rachas negativas
                
                cs = 1             # Se reinicia el contador para revisar la siguiente racha
            
            dicto = {'p':positives, 'n':negatives, 't':totales} # Almaceno los resultados en un mismo diccionario
            
            self.values = dicto
    
    
    def generate(self, values):
        # Esta funcion toma una lista de rachas, prepara el histograma y calcula las metricas
        
        # Transforma la lista de rachas en una serie de pandas 
        serie = self.pd.Series(values)
        
        # Con value counts, se cuentas cuantas rachas hubo de cada duracion (10 de 1 dia, 8 de 2 dias, 5 de 3 dias...)
        hist = serie.value_counts().to_frame()      
        hist.reset_index(drop=False,inplace=True)
        
        # mi: valor de la duracion (1 dia, 2 dias, 3 dias...)
        # ni: frecuencia de la duracion (10 veces, 8 veces, 5 veces...)
        hist.columns = ['mi','ni']   

        # Se ordena la tabla de frecuencias
        hist.sort_values(by='mi',ascending=True,inplace=True)
        
        # Se calcula el total N de datos (rachas)  
        N = int(hist['ni'].sum())
        
        # fi: Frecuencias relativas (valores normalizados por el total)
        hist['fi'] = hist['ni']/N
        
        # Calculo de la media del histograma
        stads = hist.copy()
        stads['mi*ni'] = stads['mi']*stads['ni']
        MEAN = self.np.round(sum(stads['mi*ni'])/N,3) #Promedio

        # Calculo de la desviacion estandar del histograma
        stads['mi-u'] = stads['mi'] - MEAN
        stads['(mi-u)^2'] = stads['mi-u']**2
        stads['ni*(mi-u)^2'] = stads['ni']*stads['(mi-u)^2']
        STD = self.np.round((sum(stads['ni*(mi-u)^2'])/(N-1))**0.5,3) #Desviacion
        
        # Calculo coeficiente de variacion
        CV = self.np.round(STD/MEAN,3) #Dispersion
        
        # Doy formato al histograma
        hist.columns = ['magnitud','veces','proba']
        hist.sort_index(inplace=True)
        
        # Guardo en un diccionario el histograma y las metricas calculadas
        dicto = {'h':hist,'m':MEAN,'s':STD,'cv':CV,'n':N}
        
        return dicto
        
    def get_histograms(self):
        
        keys = ['t','p','n']
        
        # Genera los histogramas
        dictos = {}
        for k in keys:
            dictos[k] = self.generate(self.values[k])
        
        # Guarda los histogramas
        histograms = {}
        for k in keys:
            histograms[k] = dictos[k]['h']
            
        # Genera un Dataframe con las metricas
        dfs = []
        for k in keys:
            dicto = dictos[k]
            df = self.pd.DataFrame([dicto['m'],dicto['s'],dicto['cv'],dicto['n']]).T
            df.columns = ['h'+k+'_mean','h'+k+'_std','h'+k+'_cv','h'+k+'_n']
            dfs.append(df)
        
        dfh = dfs[0]
        for d in dfs[1:]:
            dfh = self.pd.concat([dfh,d],axis=1)
            
        self.histograms = histograms
        self.hmetrics = dfh