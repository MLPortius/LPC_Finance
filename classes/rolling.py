# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 20:47:28 2023

@author: Andriu
"""

class RollingLPC:
    
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
        self.mse_dir = None

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
        self.msedh = None
        self.DAH = None
        
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
        
        pred_list = []
        mean_list = []
        std_list = []

        #mean_rs = [] 
        intra_p_l = []
        derivada_pred = []
        derivada_plus = []
        menos = []

        for ven in self.wlist:
            ven.predict()
            pred_list.append(ven.pred)
            mean_list.append(ven.mean)
            std_list.append(ven.std)
            
            # REVISAR QUE ES ESTO -------------------------------------
            intra_p_l.append(ven.predicted)
            derivada_pred.append(ven.predicted[-2]-ven.predicted[-3])
            derivada_plus.append(ven.predicted[-1]-ven.predicted[-2])
            menos.append(ven.dif)
            #mean_rs.append(ven.mean_rs)
        
        self.intra_pred_list = intra_p_l
        
            # ---------------------------------------------------------


        last = pred_list[-1]
        self.last = last

        pred_vector = self.np.asarray(pred_list[:-1])
        mean_vector = self.np.asarray(mean_list[:-1])
        std_vector = self.np.asarray(std_list[:-1])
        
        # REVISAR QUE ES ESTO -----------------------------------------
        deriv_vector =  self.np.asarray(derivada_pred[:-1])
        deriv_plus_vector = self.np.asarray(derivada_plus[:-1])
        menos_vector =  self.np.asarray(menos[:-1])
        # -------------------------------------------------------------


        real_vector = self.np.asarray(self.signal)
        error_vector = real_vector[self.wsize:] - pred_vector
        
        #nuevo
        deriv_df = self.pd.DataFrame(deriv_vector)
        deriv_df.index = pred_ind
        deriv_df.columns = ['DERIVADA PRED N']

        deriv_plus_df = self.pd.DataFrame(deriv_plus_vector)
        deriv_plus_df.index = pred_ind
        deriv_plus_df.columns = ['DERIVADA PRED N+1']

        menos_df = self.pd.DataFrame(menos_vector)
        menos_df.index = pred_ind
        menos_df.columns = ['X[n]-X[n-1]']


        pred_df = self.pd.DataFrame(pred_vector)
        pred_df.index = pred_ind
        pred_df.columns = ['PREDICTION']
        
        mean_df = self.pd.DataFrame(mean_vector)
        mean_df.index = pred_ind
        mean_df.columns = ['MEANS']
        
        std_df = self.pd.DataFrame(std_vector)
        std_df.index = pred_ind
        std_df.columns = ['STD']


        
        error_df = self.pd.DataFrame(error_vector)
        error_df.index = list(pred_df.index)
        error_df.columns = ['ERROR']

      
        real_df = self.pd.DataFrame(self.signal)
        real_df.columns = ['SIGNAL']

        
        
        res = self.pd.concat([real_df,pred_df,error_df,mean_df,std_df, deriv_df, deriv_plus_df, menos_df],axis=1)
        
        #print(res['SIGNAL'], res['MEANS'].isna().sum(), res['STD'].isna().sum())


        if self.normalized:

            res['NORM_SIGNAL'] = (res['SIGNAL']-res['MEANS'])/res['STD']
            res['NORM_PREDICTION'] = (res['PREDICTION']-res['MEANS'])/res['STD'] 
            res['NORM_ERROR'] = res['NORM_SIGNAL']-res['NORM_PREDICTION']
       
        self.results = res

    
    def get_metrics(self):
       
        
        # INICIALIZAR METRICAS -----------------------------------
        DF = self.results
        DF.dropna(axis=0,inplace=True)

        ytrue = DF['NORM_SIGNAL']
        ypred = DF['NORM_PREDICTION']
        # --------------------------------------------------------
        
        
        
        # REVISAR QUE ES ESTO ------------------------------------
        #Nuevo n+1
        res = self.results
        x_xn_1 = self.np.append([0],self.np.diff(res['NORM_SIGNAL']))
        xn_1_df = self.pd.DataFrame(x_xn_1)
        #print(len(xn_1_df))
        xn_1_df.index = res.index
        xn_1_df.columns = ['X[n+1]-X[n]']
        res['X[n+1]-X[n]'] = xn_1_df
        self.results = res
        # ---------------------------------------------------------



        # REVISAR QUE ES ESTO ------------------------------------
        # Nuevo coef corr
        res = self.results
        
        corr_coef_deriv_menos = self.np.corrcoef(res['DERIVADA PRED N'], res['X[n]-X[n-1]'])[0][1]
        corr_coef_deriv_mas = self.np.corrcoef(res['DERIVADA PRED N'], res['X[n+1]-X[n]'])[0][1]
        corr_coef_orig = self.np.corrcoef(res['X[n]-X[n-1]'], res['X[n+1]-X[n]'])[0][1]
        corr_coef_pred = self.np.corrcoef(res['DERIVADA PRED N+1'], res['DERIVADA PRED N'])[0][1]
        corr_coef_plus =  self.np.corrcoef(res['DERIVADA PRED N+1'], res['X[n+1]-X[n]'])[0][1]

        self.cc_dme = corr_coef_deriv_menos
        self.cc_dma =corr_coef_deriv_mas
        self.cc_o = corr_coef_orig 
        self.cc_pred = corr_coef_pred
        self.cc_plus = corr_coef_plus
        
        # NUEVO CROSS CORRELATION
        # ---------------------------------------------------------
        
        
        
        #NUEVO DH y DAH -------------------------------------------
        df = self.pd.concat([ytrue,ypred],axis=1)
        df.columns = ['true','pred']
        df['r_true'] = df['true'].shift(1)
        df['r_pred'] = df['pred'].shift(1)
        df['d_true'] = df['true']-df['r_true']
        df['d_pred'] = df['pred']-df['r_pred']
        df=df.reset_index()
        umbral = 0.2
        for i in range(1,len(df)):
            if df['d_true'][i]>umbral:
                df['d_true'][i] = 1
            elif df['d_true'][i]<=umbral and df['d_true'][i]>=-umbral:
                df['d_true'][i] = 0
            else:
                df['d_true'][i] = -1
            if df['d_pred'][i]>=umbral:
                df['d_pred'][i] = 1
            elif df['d_pred'][i]<=umbral and df['d_pred'][i]>=-umbral:
                df['d_pred'][i] = 0
            else:
                df['d_pred'][i] = -1
                  
        df_mini2 = df[['d_true','d_pred']]
        df_mini2 = df_mini2.dropna()#resetindex

        mse_dh = sum((df_mini2['d_true']-df_mini2['d_pred'])**2)/len(df_mini2)
        self.msedh = mse_dh

        dah = sum(df_mini2['d_true'] == df_mini2['d_pred'])/len(df_mini2)
        self.DAH =dah
        # ----------------------------------------------------------

        
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
        mse_dir = self.mse_dir
        
        # Analisis de Rs LPC
        # desv_rs = self.nm_std_r
        # mstd_r = self.mstd_rs
        # mstd_rn = self.mstd_rs2

        # Analisis con Histeresis
        msedh = self.msedh
        DAH = self.DAH
        
        # Analisis Derivadas y Cross Correlation
        #mean_f = self.mean_f
        cc_dme = self.cc_dme
        cc_dma = self.cc_dma 
        cc_o = self.cc_o 
        cc_pred = self.cc_pred 
        cc_plus = self.cc_plus
        
        
        # Create Rolling Summary
        summ = [n, ws, p, norm, last, mae, da, dao, dis, rmse_dir,
                msedh, DAH, cc_dme, cc_dma, cc_o, cc_pred, cc_plus]

        df = self.pd.DataFrame(summ)
        df = df.transpose()
        df.columns = ['n_data','w_size','p_lags','normalized','last_pred',
                      'MAE','DA','DA_Oracle','DIS','RMSE_dir','MSE_DH','DAH',
                      'CorrCoef_d_minus','CorrCoef_d_plus','CorrCoef_original_plus_m',
                      'CorrCoef_pred(n,n+1)','CorrCoef_deriv_orig(n+1)']
        df.index = [label]

        return df


    def clean(self):
        self.signal = 'cleaned'
        self.wlist = 'cleaned'
        
        res = self.results
        res_new = res.loc[:,['NORM_SIGNAL','NORM_PREDICTION','MEANS','STD']]
        self.results = res_new
    