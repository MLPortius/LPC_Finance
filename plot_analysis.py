# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 17:19:58 2023

@author: Andriu
"""

#%% IMPORT LIBRARIES

import argparse
import numpy as np
import pandas as pd
import plotly.express as px
import compress_pickle as cpickle
import plotly.graph_objects as go
from scipy.interpolate import griddata

#%% SCRIPT ARGUMENTS

parser = argparse.ArgumentParser(prog= 'LPC_Plot_Analysis',
                                 description = 'Analyze results and Plot metrics...',
                                 epilog = 'Created by Andriu')

parser.add_argument('-d','--dset', required=True, type=int, choices=[1,2,3])
args = parser.parse_args()

# ARGUMENTS
ds = args.dset

#%% SCRIPT SETUP
afolder = 'output/analysis/'
pfolder = 'output/plots/'

dset_path = 'set'+str(ds)+'/'

dates = {}
dates['1'] = {'s':'2010-09-08', 'e':'2016-08-22'}
dates['2'] = {'s':'2016-08-23', 'e':'2022-08-08'}
dates['3'] = {'s':'2010-09-08', 'e':'2022-08-08'}

sample = '['+dates[str(ds)]['s'] + ' : ' +  dates[str(ds)]['e']+']'

dpi = 500
figsize = (2,1)

#%% LOAD DATA

with open(afolder+dset_path+'dfs.lzma','rb') as file:
  data = cpickle.load(file)
del(file)

da = data['da']
mae = data['mae']
dis = data['dis']

da['stock'] = list(da.index)
mae['stock'] = list(mae.index)
dis['stock'] = list(dis.index)

mae.columns = [x.upper() for x in list(mae.columns)]
dis.columns = [x.upper() for x in list(dis.columns)]
da.columns = [x.upper() for x in list(da.columns)]

#%% GRID OPTIMIZATION

# ----------------------- SURFACE: MAE vs P vs W (MIN MAE) --------------------
theme = 'GRID OPTIMIZATION ...'
print(theme,'(',1,'/',3,')')

file_title = 'SURFACE_MAE_P_W'
file_type = 'grid_optimization'

df = mae

zvar = ['MAE','Mean Absolute Error (MAE)']
yvar = ['W_SIZE','Window Size (W)']
xvar = ['P_LAGS','Lags (P)']

x = np.array(df[xvar[0]])
y = np.array(df[yvar[0]])
z = np.array(df[zvar[0]])

i=list(df.index)

xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)

X,Y = np.meshgrid(xi,yi)
Z = griddata((x,y),z,(X,Y), method='cubic')

fig = go.Figure()
fig.add_trace(go.Surface(x=xi,y=yi,z=Z,customdata= np.stack((x, y, z), axis=-1),hovertemplate = 'P_LAGS: %{x:.0f}'+\
                '<br>W_SIZE: %{y:.0f}'+\
                '<br>MAE: %{z:.3f}<extra></extra>'))

t = 'MAE vs P vs W - MIN MAE - '+sample


fig.update_layout(title=t,scene = dict(
                    xaxis_title=xvar[1],
                    yaxis_title=yvar[1],
                    zaxis_title=zvar[1]),
                    width=figsize[0]*dpi,
                    height=figsize[1]*dpi)

fig.write_image(pfolder+dset_path+file_type+'/'+'png/'+file_title+'.png')
fig.write_image(pfolder+dset_path+file_type+'/'+'svg/'+file_title+'.svg')
fig.write_html(pfolder+dset_path+file_type+'/'+'html/'+file_title+'.html')


# ----------------------- SURFACE: DA vs P vs W (MAX DA) ----------------------
theme = 'GRID OPTIMIZATION ...'
print(theme,'(',2,'/',3,')')

file_title = 'SURFACE_DA_P_W'
file_type = 'grid_optimization'

df = da

zvar = ['DA','Directional Accuracy (DA)']
yvar = ['W_SIZE','Window Size (W)']
xvar = ['P_LAGS','Lags (P)']

x = np.array(df[xvar[0]])
y = np.array(df[yvar[0]])
z = np.array(df[zvar[0]])

i=list(df.index)

xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)

X,Y = np.meshgrid(xi,yi)
Z = griddata((x,y),z,(X,Y), method='cubic')

fig = go.Figure()
fig.add_trace(go.Surface(x=xi,y=yi,z=Z,customdata= np.stack((x, y, z), axis=-1),hovertemplate = 'P_LAGS: %{x:.0f}'+\
                '<br>W_SIZE: %{y:.0f}'+\
                '<br>DA: %{z:.3f}<extra></extra>'))

t = 'DA vs P vs W - MAX DA - '+sample


fig.update_layout(title=t,scene = dict(
                    xaxis_title=xvar[1],
                    yaxis_title=yvar[1],
                    zaxis_title=zvar[1]),
                    width=figsize[0]*dpi,
                    height=figsize[1]*dpi)

fig.write_image(pfolder+dset_path+file_type+'/'+'png/'+file_title+'.png')
fig.write_image(pfolder+dset_path+file_type+'/'+'svg/'+file_title+'.svg')
fig.write_html(pfolder+dset_path+file_type+'/'+'html/'+file_title+'.html')

# ----------------------- SURFACE: DIS vs P vs W (MIN DIS) --------------------
theme = 'GRID OPTIMIZATION ...'
print(theme,'(',3,'/',3,')')

file_title = 'SURFACE_DIS_P_W'
file_type = 'grid_optimization'

df = dis

zvar = ['DIS','Autocorrelation Discriminant (DIS)']
yvar = ['W_SIZE','Window Size (W)']
xvar = ['P_LAGS','Lags (P)']

x = np.array(df[xvar[0]])
y = np.array(df[yvar[0]])
z = np.array(df[zvar[0]])

i=list(df.index)

xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)

X,Y = np.meshgrid(xi,yi)
Z = griddata((x,y),z,(X,Y), method='cubic')

fig = go.Figure()
fig.add_trace(go.Surface(x=xi,y=yi,z=Z,customdata= np.stack((x, y, z), axis=-1),hovertemplate = 'P_LAGS: %{x:.0f}'+\
                '<br>W_SIZE: %{y:.0f}'+\
                '<br>DIS: %{z:.3f}<extra></extra>'))

t = 'DIS vs P vs W - MIN DIS - '+sample


fig.update_layout(title=t,scene = dict(
                    xaxis_title=xvar[1],
                    yaxis_title=yvar[1],
                    zaxis_title=zvar[1]),
                    width=figsize[0]*dpi,
                    height=figsize[1]*dpi)

fig.write_image(pfolder+dset_path+file_type+'/'+'png/'+file_title+'.png')
fig.write_image(pfolder+dset_path+file_type+'/'+'svg/'+file_title+'.svg')
fig.write_html(pfolder+dset_path+file_type+'/'+'html/'+file_title+'.html')


#%% METRICS RELATION

# ----------------------- SCATTER: DA vs MAE (MAX DA) -------------------------
theme = 'METRICS RELATION ...'
print(theme,'(',1,'/',3,')')

file_title = 'SCATTER_DA_MAE'
file_type = 'metrics_relation'

yvar = ['DA','Directional Accuracy (DA)']
xvar = ['MAE','Mean Absolute Error (MAE)']

t = 'DA vs MAE - MAX DA - '+sample
c = 'royalblue'

fig = px.scatter(da, y=yvar[0], x=xvar[0], title=t, trendline='ols',color_discrete_sequence=[c],
                 width=figsize[0]*dpi, height=figsize[1]*dpi, hover_data=['STOCK'])
fig.update_layout(plot_bgcolor='white',showlegend=False)
fig.update_xaxes(showline=False,linecolor='black',showgrid=False,gridcolor='lightgrey',title=xvar[1])
fig.update_yaxes(showline=False,linecolor='black',showgrid=True,gridcolor='lightgrey',title=yvar[1])
fig.show()

fig.write_image(pfolder+dset_path+file_type+'/'+'png/'+file_title+'.png')
fig.write_image(pfolder+dset_path+file_type+'/'+'svg/'+file_title+'.svg')
fig.write_html(pfolder+dset_path+file_type+'/'+'html/'+file_title+'.html')

# ----------------------- SCATTER: DA vs DIS (MAX DA) -------------------------
theme = 'METRICS RELATION ...'
print(theme,'(',2,'/',3,')')

file_title = 'SCATTER_DA_DIS'
file_type = 'metrics_relation'

yvar = ['DA','Directional Accuracy (DA)']
xvar = ['DIS','Autocorrelation Discriminant (DIS)']

t = 'DA vs DIS - MAX DA - '+sample
c = 'royalblue'

fig = px.scatter(da, y=yvar[0], x=xvar[0], title=t, trendline='ols',color_discrete_sequence=[c],
                 width=figsize[0]*dpi, height=figsize[1]*dpi, hover_data=['STOCK'])
fig.update_layout(plot_bgcolor='white',showlegend=False)
fig.update_xaxes(showline=False,linecolor='black',showgrid=False,gridcolor='lightgrey',title=xvar[1])
fig.update_yaxes(showline=False,linecolor='black',showgrid=True,gridcolor='lightgrey',title=yvar[1])
fig.show()

fig.write_image(pfolder+dset_path+file_type+'/'+'png/'+file_title+'.png')
fig.write_image(pfolder+dset_path+file_type+'/'+'svg/'+file_title+'.svg')
fig.write_html(pfolder+dset_path+file_type+'/'+'html/'+file_title+'.html')

# ----------------------- SCATTER: MAE vs DIS (MAX DA) ------------------------
theme = 'METRICS RELATION ...'
print(theme,'(',3,'/',3,')')

file_title = 'SCATTER_MAE_DIS'
file_type = 'metrics_relation'

yvar = ['MAE','Mean Absolute Error (MAE)']
xvar = ['DIS','Autocorrelation Discriminant (DIS)']

t = 'MAE vs DIS - MAX DA - '+sample
c = 'royalblue'

fig = px.scatter(da, y=yvar[0], x=xvar[0], title=t, trendline='ols',color_discrete_sequence=[c],
                 width=figsize[0]*dpi, height=figsize[1]*dpi, hover_data=['STOCK'])
fig.update_layout(plot_bgcolor='white',showlegend=False)
fig.update_xaxes(showline=False,linecolor='black',showgrid=False,gridcolor='lightgrey',title=xvar[1])
fig.update_yaxes(showline=False,linecolor='black',showgrid=True,gridcolor='lightgrey',title=yvar[1])
fig.show()

fig.write_image(pfolder+dset_path+file_type+'/'+'png/'+file_title+'.png')
fig.write_image(pfolder+dset_path+file_type+'/'+'svg/'+file_title+'.svg')
fig.write_html(pfolder+dset_path+file_type+'/'+'html/'+file_title+'.html')

#%% DIRECTIONAL ACCURACY ORACLE

# ----------------------- SCATTER: DA vs DA_ORACLE (MAX DA) -------------------
theme = 'DIRECTIONAL ACCURACY ORACLE ...'
print(theme,'(',1,'/',1,')')

file_title = 'SCATTER_DA_DAO'
file_type = 'da_oracle_analysis'

yvar = ['DA','Directional Accuracy (DA)']
xvar = ['DA_ORACLE','Directional Accuracy Oracle (DAO)']

t = 'DA vs DAO - MAX DA - '+sample
c = 'salmon'

fig = px.scatter(da, y=yvar[0], x=xvar[0], title=t, trendline='ols',color_discrete_sequence=[c],
                 width=figsize[0]*dpi, height=figsize[1]*dpi, hover_data=['STOCK'])
fig.update_layout(plot_bgcolor='white',showlegend=False)
fig.update_xaxes(showline=False,linecolor='black',showgrid=False,gridcolor='lightgrey',title=xvar[1])
fig.update_yaxes(showline=False,linecolor='black',showgrid=True,gridcolor='lightgrey',title=yvar[1])
fig.show()

fig.write_image(pfolder+dset_path+file_type+'/'+'png/'+file_title+'.png')
fig.write_image(pfolder+dset_path+file_type+'/'+'svg/'+file_title+'.svg')
fig.write_html(pfolder+dset_path+file_type+'/'+'html/'+file_title+'.html')

#%% BINARY CURVES

# ----------------------- SCATTER: DA vs RMSE_DIR (MAX DA) --------------------
theme = 'BINARY CURVES ...'
print(theme,'(',1,'/',1,')')

file_title = 'SCATTER_DA_RMSE_DIR'
file_type = 'binary_curves_analysis'

yvar = ['DA','Directional Accuracy (DA)']
xvar = ['RMSE_DIR','Directional Root Mean Squared Error (RMSE_DIR)']

t = 'DA vs RMSE_DIR - MAX DA - '+sample
c = 'seagreen'

fig = px.scatter(da, y=yvar[0], x=xvar[0], title=t, trendline='ols',color_discrete_sequence=[c],
                 width=figsize[0]*dpi, height=figsize[1]*dpi, hover_data=['STOCK'])
fig.update_layout(plot_bgcolor='white',showlegend=False)
fig.update_xaxes(showline=False,linecolor='black',showgrid=False,gridcolor='lightgrey',title=xvar[1])
fig.update_yaxes(showline=False,linecolor='black',showgrid=True,gridcolor='lightgrey',title=yvar[1])
fig.show()

fig.write_image(pfolder+dset_path+file_type+'/'+'png/'+file_title+'.png')
fig.write_image(pfolder+dset_path+file_type+'/'+'svg/'+file_title+'.svg')
fig.write_html(pfolder+dset_path+file_type+'/'+'html/'+file_title+'.html')

#%% VOLUME ANALYSIS

# ----------------------- SCATTER: DA vs VSUM (MAX DA) ------------------------
theme = 'VOLUME ANALYSIS ...'
print(theme,'(',1,'/',4,')')

file_title = 'SCATTER_DA_VSUM'
file_type = 'volume_analysis'

yvar = ['DA','Directional Accuracy (DA)']
xvar = ['VSUM','Natural Logarithm of Total Volume (lnVSUM)']

t = 'DA vs ln(VSUM) - MAX DA - '+sample
c = 'forestgreen'

df = da.copy()
df[xvar[0]] = np.log(df[xvar[0]])

fig = px.scatter(df, y=yvar[0], x=xvar[0], title=t, trendline='ols',color_discrete_sequence=[c],
                 width=figsize[0]*dpi, height=figsize[1]*dpi, hover_data=['STOCK'])
fig.update_layout(plot_bgcolor='white',showlegend=False)
fig.update_xaxes(showline=False,linecolor='black',showgrid=False,gridcolor='lightgrey',title=xvar[1])
fig.update_yaxes(showline=False,linecolor='black',showgrid=True,gridcolor='lightgrey',title=yvar[1])
fig.show()

fig.write_image(pfolder+dset_path+file_type+'/'+'png/'+file_title+'.png')
fig.write_image(pfolder+dset_path+file_type+'/'+'svg/'+file_title+'.svg')
fig.write_html(pfolder+dset_path+file_type+'/'+'html/'+file_title+'.html')

# ----------------------- SCATTER: DA vs VCV (MAX DA) -------------------------
theme = 'VOLUME ANALYSIS ...'
print(theme,'(',2,'/',4,')')

file_title = 'SCATTER_DA_VCV'
file_type = 'volume_analysis'

yvar = ['DA','Directional Accuracy (DA)']
xvar = ['VCV','Natural Logarithm of Volume Coefficient of Variation (lnVCV)']

t = 'DA vs lnVCV - MAX DA - '+sample
c = 'forestgreen'

df = da.copy()
df[xvar[0]] = np.log(df[xvar[0]])

fig = px.scatter(df, y=yvar[0], x=xvar[0], title=t, trendline='ols',color_discrete_sequence=[c],
                 width=figsize[0]*dpi, height=figsize[1]*dpi, hover_data=['STOCK'])
fig.update_layout(plot_bgcolor='white',showlegend=False)
fig.update_xaxes(showline=False,linecolor='black',showgrid=False,gridcolor='lightgrey',title=xvar[1])
fig.update_yaxes(showline=False,linecolor='black',showgrid=True,gridcolor='lightgrey',title=yvar[1])
fig.show()

fig.write_image(pfolder+dset_path+file_type+'/'+'png/'+file_title+'.png')
fig.write_image(pfolder+dset_path+file_type+'/'+'svg/'+file_title+'.svg')
fig.write_html(pfolder+dset_path+file_type+'/'+'html/'+file_title+'.html')

# ----------------------- SCATTER: MAE vs VCV (MAX DA) ------------------------
theme = 'VOLUME ANALYSIS ...'
print(theme,'(',3,'/',4,')')

file_title = 'SCATTER_MAE_VCV'
file_type = 'volume_analysis'

yvar = ['MAE','Mean Absolute Error (MAE)']
xvar = ['VCV','Natural Logarithm Volume Coefficient of Variation (lnVCV)']

t = 'MAE vs lnVCV - MAX DA - '+sample
c = 'forestgreen'

df = da.copy()
df[xvar[0]] = np.log(df[xvar[0]])


fig = px.scatter(df, y=yvar[0], x=xvar[0], title=t, trendline='ols',color_discrete_sequence=[c],
                 width=figsize[0]*dpi, height=figsize[1]*dpi, hover_data=['STOCK'])
fig.update_layout(plot_bgcolor='white',showlegend=False)
fig.update_xaxes(showline=False,linecolor='black',showgrid=False,gridcolor='lightgrey',title=xvar[1])
fig.update_yaxes(showline=False,linecolor='black',showgrid=True,gridcolor='lightgrey',title=yvar[1])
fig.show()

fig.write_image(pfolder+dset_path+file_type+'/'+'png/'+file_title+'.png')
fig.write_image(pfolder+dset_path+file_type+'/'+'svg/'+file_title+'.svg')
fig.write_html(pfolder+dset_path+file_type+'/'+'html/'+file_title+'.html')

# ----------------------- SCATTER: DIS vs VCV (MAX DA) ------------------------
theme = 'VOLUME ANALYSIS ...'
print(theme,'(',4,'/',4,')')

file_title = 'SCATTER_DIS_VCV'
file_type = 'volume_analysis'

yvar = ['DIS','Autocorrelation Discriminant (DIS)']
xvar = ['VCV','Natural Logarithn of Volume Coefficient of Variation (lnVCV)']

t = 'DIS vs lnVCV - MAX DA - '+sample
c = 'forestgreen'

df = da.copy()
df[xvar[0]] = np.log(df[xvar[0]])


fig = px.scatter(df, y=yvar[0], x=xvar[0], title=t, trendline='ols',color_discrete_sequence=[c],
                 width=figsize[0]*dpi, height=figsize[1]*dpi, hover_data=['STOCK'])
fig.update_layout(plot_bgcolor='white',showlegend=False)
fig.update_xaxes(showline=False,linecolor='black',showgrid=False,gridcolor='lightgrey',title=xvar[1])
fig.update_yaxes(showline=False,linecolor='black',showgrid=True,gridcolor='lightgrey',title=yvar[1])
fig.show()

fig.write_image(pfolder+dset_path+file_type+'/'+'png/'+file_title+'.png')
fig.write_image(pfolder+dset_path+file_type+'/'+'svg/'+file_title+'.svg')
fig.write_html(pfolder+dset_path+file_type+'/'+'html/'+file_title+'.html')

#%% HISTERESIS BINARY CURVES

# ----------------------- SCATTER: RMSE_DH vs DAH (MAX DA) --------------------
theme = 'HISTERESIS BINARY CURVES ...'
print(theme,'(',1,'/',2,')')

file_title = 'SCATTER_RMSEDH_DAH'
file_type = 'histeresis_binary_curves_analysis'

yvar = ['RMSE_DH','Histeresis Directional RMSE (RMSE_DH)']
xvar = ['DAH','Histeresis Directional Accuracy (DAH)']

t = 'RMSE_DH vs DAH - MAX DA - '+sample
c = 'slategray'

fig = px.scatter(da, y=yvar[0], x=xvar[0], title=t, trendline='ols',color_discrete_sequence=[c],
                 width=figsize[0]*dpi, height=figsize[1]*dpi, hover_data=['STOCK'])
fig.update_layout(plot_bgcolor='white',showlegend=False)
fig.update_xaxes(showline=False,linecolor='black',showgrid=False,gridcolor='lightgrey',title=xvar[1])
fig.update_yaxes(showline=False,linecolor='black',showgrid=True,gridcolor='lightgrey',title=yvar[1])
fig.show()

fig.write_image(pfolder+dset_path+file_type+'/'+'png/'+file_title+'.png')
fig.write_image(pfolder+dset_path+file_type+'/'+'svg/'+file_title+'.svg')
fig.write_html(pfolder+dset_path+file_type+'/'+'html/'+file_title+'.html')

# ----------------------- SCATTER: DA vs DAH (MAX DA) -------------------------
theme = 'HISTERESIS BINARY CURVES ...'
print(theme,'(',2,'/',2,')')

file_title = 'SCATTER_DA_DAH'
file_type = 'histeresis_binary_curves_analysis'

yvar = ['DA','Directional Accuracy (DA)']
xvar = ['DAH','Histeresis Directional Accuracy (DAH)']

t = 'DA vs DAH - MAX DA - '+sample
c = 'slategray'

fig = px.scatter(da, y=yvar[0], x=xvar[0], title=t, trendline='ols',color_discrete_sequence=[c],
                 width=figsize[0]*dpi, height=figsize[1]*dpi, hover_data=['STOCK'])
fig.update_layout(plot_bgcolor='white',showlegend=False)
fig.update_xaxes(showline=False,linecolor='black',showgrid=False,gridcolor='lightgrey',title=xvar[1])
fig.update_yaxes(showline=False,linecolor='black',showgrid=True,gridcolor='lightgrey',title=yvar[1])
fig.show()

fig.write_image(pfolder+dset_path+file_type+'/'+'png/'+file_title+'.png')
fig.write_image(pfolder+dset_path+file_type+'/'+'svg/'+file_title+'.svg')
fig.write_html(pfolder+dset_path+file_type+'/'+'html/'+file_title+'.html')

#%% STREAK HISTOGRAMS ANALYSIS

# ----------------------- SCATTER: Positive Streak Mean vs DA (MAX DA) --------
theme = 'STREAK HISTOGRAMS ANALYSIS ...'
print(theme,'(',1,'/',5,')')

file_title = 'SCATTER_DA_HPMEAN'
file_type = 'streak_histogram_analysis'

yvar = ['DA','Directional Accuracy (DA)']
xvar = ['HP_MEAN','Positive Streaks Histogram Mean (HP_MEAN)']

t = 'DA vs HP_MEAN - MAX DA - '+sample
c = 'blueviolet'

fig = px.scatter(da, y=yvar[0], x=xvar[0], title=t, trendline='ols',color_discrete_sequence=[c],
                 width=figsize[0]*dpi, height=figsize[1]*dpi, hover_data=['STOCK'])
fig.update_layout(plot_bgcolor='white',showlegend=False)
fig.update_xaxes(showline=False,linecolor='black',showgrid=False,gridcolor='lightgrey',title=xvar[1])
fig.update_yaxes(showline=False,linecolor='black',showgrid=True,gridcolor='lightgrey',title=yvar[1])
fig.show()

fig.write_image(pfolder+dset_path+file_type+'/'+'png/'+file_title+'.png')
fig.write_image(pfolder+dset_path+file_type+'/'+'svg/'+file_title+'.svg')
fig.write_html(pfolder+dset_path+file_type+'/'+'html/'+file_title+'.html')

# ----------------------- SCATTER: Negative Streak Mean vs DA (MAX DA) --------
theme = 'STREAK HISTOGRAMS ANALYSIS ...'
print(theme,'(',2,'/',5,')')


file_title = 'SCATTER_DA_HNMEAN'
file_type = 'streak_histogram_analysis'

yvar = ['DA','Directional Accuracy (DA)']
xvar = ['HN_MEAN','Negative Streaks Histogram Mean (HN_MEAN)']

t = 'DA vs HN_MEAN - MAX DA - '+sample
c = 'blueviolet'

fig = px.scatter(da, y=yvar[0], x=xvar[0], title=t, trendline='ols',color_discrete_sequence=[c],
                 width=figsize[0]*dpi, height=figsize[1]*dpi, hover_data=['STOCK'])
fig.update_layout(plot_bgcolor='white',showlegend=False)
fig.update_xaxes(showline=False,linecolor='black',showgrid=False,gridcolor='lightgrey',title=xvar[1])
fig.update_yaxes(showline=False,linecolor='black',showgrid=True,gridcolor='lightgrey',title=yvar[1])
fig.show()

fig.write_image(pfolder+dset_path+file_type+'/'+'png/'+file_title+'.png')
fig.write_image(pfolder+dset_path+file_type+'/'+'svg/'+file_title+'.svg')
fig.write_html(pfolder+dset_path+file_type+'/'+'html/'+file_title+'.html')

# ----------------------- SCATTER: Total Streak Mean vs DA (MAX DA) -----------
theme = 'STREAK HISTOGRAMS ANALYSIS ...'
print(theme,'(',3,'/',5,')')


file_title = 'SCATTER_DA_HTMEAN'
file_type = 'streak_histogram_analysis'

yvar = ['DA','Directional Accuracy (DA)']
xvar = ['HT_MEAN','Total Streaks Histogram Mean (HT_MEAN)']

t = 'DA vs HT_MEAN - MAX DA - '+sample
c = 'blueviolet'

fig = px.scatter(da, y=yvar[0], x=xvar[0], title=t, trendline='ols',color_discrete_sequence=[c],
                 width=figsize[0]*dpi, height=figsize[1]*dpi, hover_data=['STOCK'])
fig.update_layout(plot_bgcolor='white',showlegend=False)
fig.update_xaxes(showline=False,linecolor='black',showgrid=False,gridcolor='lightgrey',title=xvar[1])
fig.update_yaxes(showline=False,linecolor='black',showgrid=True,gridcolor='lightgrey',title=yvar[1])
fig.show()

fig.write_image(pfolder+dset_path+file_type+'/'+'png/'+file_title+'.png')
fig.write_image(pfolder+dset_path+file_type+'/'+'svg/'+file_title+'.svg')
fig.write_html(pfolder+dset_path+file_type+'/'+'html/'+file_title+'.html')

# ----------------------- SCATTER: Total Streak Mean vs MAE (MAX DA) ----------
theme = 'STREAK HISTOGRAMS ANALYSIS ...'
print(theme,'(',4,'/',5,')')


file_title = 'SCATTER_MAE_HTMEAN'
file_type = 'streak_histogram_analysis'

yvar = ['MAE','Mean Absolute Error (MAE)']
xvar = ['HT_MEAN','Total Streaks Histogram Mean (HT_MEAN)']

t = 'MAE vs HT_MEAN - MAX DA - '+sample
c = 'blueviolet'

fig = px.scatter(da, y=yvar[0], x=xvar[0], title=t, trendline='ols',color_discrete_sequence=[c],
                 width=figsize[0]*dpi, height=figsize[1]*dpi, hover_data=['STOCK'])
fig.update_layout(plot_bgcolor='white',showlegend=False)
fig.update_xaxes(showline=False,linecolor='black',showgrid=False,gridcolor='lightgrey',title=xvar[1])
fig.update_yaxes(showline=False,linecolor='black',showgrid=True,gridcolor='lightgrey',title=yvar[1])
fig.show()

fig.write_image(pfolder+dset_path+file_type+'/'+'png/'+file_title+'.png')
fig.write_image(pfolder+dset_path+file_type+'/'+'svg/'+file_title+'.svg')
fig.write_html(pfolder+dset_path+file_type+'/'+'html/'+file_title+'.html')

# ----------------------- SCATTER: Total Streak Mean vs DIS (MAX DA) ----------
theme = 'STREAK HISTOGRAMS ANALYSIS ...'
print(theme,'(',5,'/',5,')')


file_title = 'SCATTER_DIS_HTMEAN'
file_type = 'streak_histogram_analysis'

yvar = ['DIS','Autocorrelation Discriminant (DIS)']
xvar = ['HT_MEAN','Total Streaks Histogram Mean (HT_MEAN)']

t = 'DIS vs HT_MEAN - MAX DA - '+sample
c = 'blueviolet'

fig = px.scatter(da, y=yvar[0], x=xvar[0], title=t, trendline='ols',color_discrete_sequence=[c],
                 width=figsize[0]*dpi, height=figsize[1]*dpi, hover_data=['STOCK'])
fig.update_layout(plot_bgcolor='white',showlegend=False)
fig.update_xaxes(showline=False,linecolor='black',showgrid=False,gridcolor='lightgrey',title=xvar[1])
fig.update_yaxes(showline=False,linecolor='black',showgrid=True,gridcolor='lightgrey',title=yvar[1])
fig.show()

fig.write_image(pfolder+dset_path+file_type+'/'+'png/'+file_title+'.png')
fig.write_image(pfolder+dset_path+file_type+'/'+'svg/'+file_title+'.svg')
fig.write_html(pfolder+dset_path+file_type+'/'+'html/'+file_title+'.html')

#%% DERIVATIVE ANALYSIS

# ----------------------- SCATTER: CORRCOEF(D[n+1,n], X[n+1,n]) vs DA (MAX DA) 
theme = 'DERIVATIVE ANALYSIS ...'
print(theme,'(',1,'/',2,')')


file_title = 'SCATTER_DA_CC_DMAS_XMAS'
file_type = 'derivative_analysis'

yvar = ['DA','Directional Accuracy (DA)']
xvar = ['CC_DMAS_XMAS','Correlation Coefficient of D[n+1,n] and X[n+1,n] (CC_DMAS_XMAS)']

t = 'DA vs CC_DMAS_XMAS - MAX DA - '+sample
c = 'tomato'

fig = px.scatter(da, y=yvar[0], x=xvar[0], title=t, trendline='ols',color_discrete_sequence=[c],
                 width=figsize[0]*dpi, height=figsize[1]*dpi, hover_data=['STOCK'])
fig.update_layout(plot_bgcolor='white',showlegend=False)
fig.update_xaxes(showline=False,linecolor='black',showgrid=False,gridcolor='lightgrey',title=xvar[1])
fig.update_yaxes(showline=False,linecolor='black',showgrid=True,gridcolor='lightgrey',title=yvar[1])
fig.show()

fig.write_image(pfolder+dset_path+file_type+'/'+'png/'+file_title+'.png')
fig.write_image(pfolder+dset_path+file_type+'/'+'svg/'+file_title+'.svg')
fig.write_html(pfolder+dset_path+file_type+'/'+'html/'+file_title+'.html')

# ----------------------- SCATTER: CORRCOEF(D[n,n-1], X[n+1,n]) vs DA (MAX DA)
theme = 'DERIVATIVE ANALYSIS ...'
print(theme,'(',2,'/',2,')')


file_title = 'SCATTER_DA_CC_DMENOS_XMAS'
file_type = 'derivative_analysis'

yvar = ['DA','Directional Accuracy (DA)']
xvar = ['CC_DMENOS_XMAS','Correlation Coefficient of D[n,n-1] and X[n+1,n] (CC_DMENOS_XMAS)']

t = 'DA vs CC_DMENOS_XMAS - MAX DA - '+sample
c = 'tomato'

fig = px.scatter(da, y=yvar[0], x=xvar[0], title=t, trendline='ols',color_discrete_sequence=[c],
                 width=figsize[0]*dpi, height=figsize[1]*dpi, hover_data=['STOCK'])
fig.update_layout(plot_bgcolor='white',showlegend=False)
fig.update_xaxes(showline=False,linecolor='black',showgrid=False,gridcolor='lightgrey',title=xvar[1])
fig.update_yaxes(showline=False,linecolor='black',showgrid=True,gridcolor='lightgrey',title=yvar[1])
fig.show()

fig.write_image(pfolder+dset_path+file_type+'/'+'png/'+file_title+'.png')
fig.write_image(pfolder+dset_path+file_type+'/'+'svg/'+file_title+'.svg')
fig.write_html(pfolder+dset_path+file_type+'/'+'html/'+file_title+'.html')

print('     ...DONE!')