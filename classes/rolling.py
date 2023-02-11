# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 20:47:28 2023

@author: Andriu
"""

class CLASS:
    
    def __init__(self, data, label='Model'):
        
        # Librerias
        import numpy as np
        import pandas as pd
        
        self.np = np
        self.pd = pd
        
        
        # Modulos
        from classes import window
        self.window = window.CLASS
        
        from classes import metrics
        self.metrics = metrics.CLASS
        
        from classes import derivative
        self.deriv = derivative.CLASS
        
        
        # Atributos Inicializacion
        self.signal = data
        self.length = len(self.signal.index)
        
        self.label = label
        
        self.wsize = None
        self.p = None
        self.n_windows = None
        
        # Atributos Control
        self.wlist = []
        self.clist = []
        
        self.normalized = False
        
        self.results = self.pd.DataFrame()
        self.last = None
        
        # Atributos Metricas
        
            # Optimizacion
        self.mae = None
        self.da = None
        self.dis = None
        
        self.time = None
        
            # Analisis oracle
        self.dao = None
        
            # Analisis binario
        self.rmse_dir = None

            # Analisis de Estacionaridad Rs y ADF
        # self.std_rs = None
        # self.std_rs2 = None
        # self.nm_std_r = None
        self.adf_test = None
        # self.mstd_rs = None
        # self.mstd_rs2 = None
        
            # Analisis de Correlacion Cruzada
        self.cross_cor0 = None
        
            # Analisis con Histeresis
        self.rmse_dh = None
        self.dah = None
        
            # nuevo2
        self.intra_pred_list = []
        self.cc_dme = None
        self.cc_dma = None
        self.cc_o = None
        
            # nuevo3
        # self.mean_f = None
        self.cc_pred = None
        self.cc_plus = None
        

    def create(self, window_size):

        data = self.signal
        ws = window_size
        
        self.wsize = ws
        
        nw = self.length-ws+1
        self.n_windows = nw
        
        wl = []
        for i in range(nw):
            serie = data.iloc[i : ws+i]
            ven = self.window(serie)
            wl.append(ven)

        self.wlist = wl

    
    def normalize(self):
        
        self.normalized = True

        for ven in self.wlist:
          ven.normalize()

        
    def fit(self, p):

        self.p = p
        
        for ven in self.wlist:
            ven.fit_coef(p)

      
    def get_coefs(self):
        
        coefs_list = []
        for ven in self.wlist:
            coefs_list.append(ven.coefs)
        
        self.clist = coefs_list


    def predict(self):
        
        pred_ind = list(self.signal.index)[self.wsize:]
        
        # Listas de Resultados
        pred_list = []
        mean_list = []
        std_list = []

        #mean_rs = []
        
        # Listas de Derivadas

        #intra_p_l = []
        derivada_men = []
        derivada_mas = []
        menos = []


        for ven in self.wlist:
            
            ven.predict()
            
            # EXTRAE PREDICCIONES Y NORMALIZACION
            pred_list.append(ven.pred)
            mean_list.append(ven.mean)
            std_list.append(ven.std)
            
            # EXTRAER DERIVADAS DE VENTANAS
            
            derivada_men.append(ven.dmen)
            derivada_mas.append(ven.dmas)
            menos.append(ven.menos)
            
            # APAGADO
            #intra_p_l.append(ven.predicted)
            #mean_rs.append(ven.mean_rs)
        
        #self.intra_pred_list = intra_p_l

        last = pred_list[-1]
        self.last = last
        
        # RESULTADOS DE PREDICCION
        real_serie = self.pd.Series(self.signal,index=list(self.signal.index))
        pred_serie = self.pd.Series(pred_list[:-1], index=pred_ind)
        mean_serie = self.pd.Series(mean_list[:-1], index=pred_ind)
        std_serie = self.pd.Series(std_list[:-1], index=pred_ind)
        
        df1 = self.pd.concat([real_serie,pred_serie,mean_serie,std_serie],axis=1)
        df1.columns = ['SIGNAL','PREDICTION','MEANS','STD']
        df1.dropna(inplace=True)
        
        # RESULTADOS DE DERIVADA
        dmen_serie = self.pd.Series(derivada_men[:-1], index=pred_ind)
        dmas_serie = self.pd.Series(derivada_mas[:-1], index=pred_ind)
        menos_serie = self.pd.Series(menos[:-1], index=pred_ind)
        
        df2 = self.pd.concat([dmen_serie,dmas_serie,menos_serie],axis=1)
        df2.columns = ['DERIVADA N', 'DERIVADA N+1', 'X[n]-X[n-1]']
        df2.dropna(inplace=True)

        # CONCATENAR RESULTADOS
        df = self.pd.concat([df1,df2],axis=1)

        if self.normalized:
            
            df['NORM_SIGNAL'] = (df['SIGNAL']-df['MEANS'])/df['STD']
            df['NORM_PREDICTION'] = (df['PREDICTION']-df['MEANS'])/df['STD'] 
            df['NORM_ERROR'] = df['NORM_SIGNAL']-df['NORM_PREDICTION']
       
        self.results = df

    
    def get_metrics(self):
       
        
        # INICIALIZAR METRICAS -----------------------------------
        DF = self.results
        DF.dropna(axis=0,inplace=True)

        ytrue = DF['NORM_SIGNAL']
        ypred = DF['NORM_PREDICTION']
        # --------------------------------------------------------
        
          
        # ANALISIS DE DERIVADAS ----------------------------------
        deriv = self.deriv(signal=DF['NORM_SIGNAL'], menos=DF['X[n]-X[n-1]'], 
                           dmas=DF['DERIVADA N+1'] , dmen=DF['DERIVADA N'])
        
        deriv_res = deriv.get_corrcoefs()
        
        self.cc_dmen_menos = deriv_res[0]
        self.cc_dmen_mas = deriv_res[1]
        self.cc_menos_mas = deriv_res[2]
        self.cc_dmas_dmen = deriv_res[3]
        self.cc_dmas_mas  = deriv_res[4]
        # ---------------------------------------------------------
        
        # CROSS CORRELATION
        
        # PREDICTIVE METRICS ---------------------------------------
        met = self.metrics(true=ytrue, pred=ypred)
        
        # Mean Absolute Error
        self.mae = met.get_mae()
        
        # Directional Accuracy
        self.da = met.get_da()
        
        # Discriminante AC
        self.dis = met.get_dis(1)
        
        # Directional Accuracy Oracle
        self.dao = met.get_oracle()
        
        # Directional RMSE
        self.rmse_dir = met.get_binary()
        
        # Directional RMSE con Histeresis, Directional Accuracy con Histeresis
        self.rmse_dh, self.dah  = met.get_histeresis(0.2)
        
        # ----------------------------------------------------------
    
    
    def summary(self):
        
        # Inicializacion
        label = self.label
        n = self.length
        
        # Control de Rolling
        norm = self.normalized
        ws = self.wsize
        p = self.p
        
        # Analisis Predictivo
        last = self.last
        
        mae = self.mae
        da = self.da
        dao = self.dao
        
        # Autocorrelaciones
        dis = self.dis
        
        # Analisis Binario
        rmse_dir = self.rmse_dir
        
        # Analisis de Rs LPC
        # desv_rs = self.nm_std_r
        # mstd_r = self.mstd_rs
        # mstd_rn = self.mstd_rs2

        # Analisis con Histeresis
        rmse_dh = self.rmse_dh
        dah = self.dah
        
        # Analisis Derivadas y Cross Correlation
        #mean_f = self.mean_f
        cc_1 = self.cc_dmen_menos
        cc_2 = self.cc_dmen_mas
        cc_3 = self.cc_menos_mas 
        cc_4 = self.cc_dmas_dmen
        cc_5 = self.cc_dmas_mas
        
        # Create Rolling Summary
        summ = [n, ws, p, norm, last, mae, da, dao, dis, rmse_dir,
                rmse_dh, dah, cc_1, cc_2, cc_3, cc_4, cc_5]

        df = self.pd.DataFrame(summ)
        df = df.transpose()
        
        df.columns = ['n_data','w_size','p_lags','normalized','last_pred',
                      'MAE','DA','DA_Oracle','DIS','RMSE_dir','RMSE_DH','DAH',
                      'CC_Dmenos_Xmenos','CC_Dmenos_Xmas','CC_Xmenos_Xmas',
                      'CC_Dmas_Dmenos','CC_Dmas_Xmas']
        
        df.index = [label]
        df._set_axis_name('stock',inplace=True)

        return df


    def clean(self):
        self.signal = 'cleaned'
        self.wlist = 'cleaned'
        
        res = self.results
        res_new = res.loc[:,['NORM_SIGNAL','NORM_PREDICTION','MEANS','STD']]
        self.results = res_new
    