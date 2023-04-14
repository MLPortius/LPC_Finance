# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 04:35:42 2023

@author: Andriu
"""


#%% IMPORT LIBRARIES

import os
import numpy as np
import pandas as pd
import compress_pickle as cpickle

import torch
from typing import Tuple

import warnings 
warnings.filterwarnings('ignore')

import argparse

import subprocess

#%% SCRIPT SETUP

parser = argparse.ArgumentParser(prog= 'LPC_cusum_detectors',
                                 description = 'Analyse a Time Serie with CUSUM BREAKS',
                                 epilog = 'Created by Andriu')

parser.add_argument('-ut', '--token', required=True, type=str)
parser.add_argument('-un', '--user', required=True, type=str)
parser.add_argument('-ue', '--email', required=True, type=str)
parser.add_argument('-r', '--repo', required=True, type=str)

parser.add_argument('-d','--dset', required=True, type=int, choices=[1,2,3])
parser.add_argument('--git', required=True, type=int, choices=[0,1])

parser.add_argument('-p','--platform',required=True, type=str, choices=['colab','spyder'])

args = parser.parse_args()


# ARGUMENTS
ds = args.dset
git = args.git
pf = args.platform

git_branch = 'cusum/set'+str(ds)


#%% ENVIRONMENT SETUP

# ds = 1

outfolder = 'input/cusum/'

dset = 'set'+str(ds)
dpath = dset+'/' 


#%% GITHUB SETUP

print('\nGithub setup ...')

git_url = args.repo

# REPO AND USER

git_user = args.user
git_email = args.email
git_token = args.token

git_host = 'MLPortius'
project = git_url.split(git_host+'/')[1]

git_remote = 'https://'+git_token+'@github.com/'+git_host+'/'+project+'.git'

# USER ACCESS
        
print('     ...accesing to repo',git_url)

os.system("git"+" "+"config --global user.name"+" "+git_user)
os.system("git"+" "+"config --global user.email"+" "+git_email)
os.system("git"+" "+"config --global user.password"+" "+git_token)

# REPO SETUP
if pf == 'spyder':
    remotes = subprocess.check_output("git remote")
    remotes = remotes.decode().split('\n')[:-1]

    if not "lpc" in remotes:
        print('     ...adding lpc remote to repo')
        os.system("git"+" "+"remote add lpc"+" "+git_remote)
        os.system("git"+" "+"pull"+" "+"lpc"+" "+"master")
        
        branches = subprocess.check_output("git branch")
        branches = branches.decode().split('\n')[:-1]
        
        branch = False
        for b in branches: 
            if git_branch in b:
                branch = True

        if not branch:
            print('     ...adding the new branch to repo')
            os.system("git"+" "+"branch"+" "+git_branch)

        os.system("git"+" "+"checkout"+" "+git_branch)
    
        print('DONE!')
        
elif pf == 'colab':
    
    print('     ...adding lpc remote to repo')
    os.system("git"+" "+"remote add lpc"+" "+git_remote)
    
    print('     ...adding the new branch to repo')
    os.system("git"+" "+"branch"+" "+git_branch)
    
    print('     ...selecting new branch')
    os.system("git"+" "+"checkout"+" "+git_branch)
    
    print('DONE!')


#%% DATA FUNCTIONS 

def GET_PRICES_DICT(i, save=False):
    
    dicto = cpickle.load('input/cpickle/lpc_datasets.lzma')

    d1 = dicto['set1']
    d2 = dicto['set2']
    
    d1 = [x['close'] for x in d1]
    d2 = [x['close'] for x in d2]
    
    df1 = d1[0]
    for d in d1[1:]:
        df1 = pd.concat([df1, d], axis=1)
    
    df2 = d2[0]
    for d in d2[1:]:
        df2 = pd.concat([df2, d], axis=1)
    
    df3 = pd.concat([df1, df2], axis=0)
    
    if save:
        dts = {'set1':df1.index, 'set2':df2.index, 'set3':df3.index}
        with open('input/info/datasets_dates.lzma', 'wb') as file:
            cpickle.dump(dts, file, compression='lzma')
    
    else:
        
        if i == 1:
            output = {}
            for c in list(df1.columns):
                output[c] = df1.loc[:,c]
        
        elif i == 2:
            output = {}
            for c in list(df2.columns):
                output[c] = df2.loc[:,c]
        
        elif i == 3:
            output = {}
            for c in list(df3.columns):
                output[c] = df3.loc[:,c]
                
        else:
            output = 'error'
            
        return output
    

def GET_NORM_PRICES(i, tickers):
    
    da = cpickle.load('output/trading_simulation/set'+str(i)+'/da_grid_results.lzma')
    mae = cpickle.load('output/trading_simulation/set'+str(i)+'/mae_grid_results.lzma')
    dis = cpickle.load('output/trading_simulation/set'+str(i)+'/dis_grid_results.lzma')
    
    norm = {}
    for t in tickers:
        norm[t] = {'da':da[t]['NORM_SIGNAL'], 
                   'mae':mae[t]['NORM_SIGNAL'], 
                   'dis':dis[t]['NORM_SIGNAL']}
    
    return norm



#%% LOAD DATA

print('\nLoading tickers...')

with open('input/info/'+'lpc_tickers.txt','r') as file:
    lpc = file.read()
    lpc = lpc.split(';')
    lpc = lpc[:-1]

print('     ...done!')


print('\nLoading prices...')

prices = GET_PRICES_DICT(ds)

print('     ...done!')


print('\nLoading norm prices...')

norm = GET_NORM_PRICES(ds, lpc)

print('     ...done!')


print('\nLoading done hursts...')

files = os.listdir(outfolder + dpath)
files.pop(files.index('.gitkeep'))

done_list = [x.split('.')[0] for x in files]

for dl in done_list:
    lpc.pop(lpc.index(dl))
    
print('     ...done!')



#%% CLASS DEFINITION

class CUSUM_DETECTOR():
        
    def __init__(self, t_warmup = 30, p_limit = 0.01) -> None:
        self._t_warmup = t_warmup
        self._p_limit = p_limit
        
        self._reset()
        
    def predict_next(self, y: torch.tensor) -> Tuple[float,bool]:
        self._update_data(y)

        if self.current_t == self._t_warmup:
            self._init_params()
        
        if self.current_t >= self._t_warmup:
            prob, is_changepoint = self._check_for_changepoint()
            if is_changepoint:
                self._reset()

            return (1-prob), is_changepoint
        
        else:
            return 0, False
            

    def _reset(self) -> None:
        self.current_t = torch.zeros(1)
                
        self.current_obs = []
        
        self.current_mean = None
        self.current_std = None
    
    def _update_data(self, y: torch.tensor) -> None:
        self.current_t += 1
        self.current_obs.append(y.reshape(1))

        
    def _init_params(self) -> None:
        self.current_mean = torch.mean(torch.cat(self.current_obs))
        self.current_std = torch.std(torch.cat(self.current_obs))
             
    
    def _check_for_changepoint(self) -> Tuple[float,bool]:
        standardized_sum = torch.sum(torch.cat(self.current_obs) - self.current_mean)/(self.current_std * self.current_t**0.5)
        prob = float(self._get_prob(standardized_sum).detach().numpy())
        
        return prob, prob < self._p_limit
    
    def _get_prob(self, y: torch.tensor) -> bool:
        p = torch.distributions.normal.Normal(0,1).cdf(torch.abs(y))
        prob = 2*(1 - p)
        
        return prob
    
    
#%% FUNCTION DEFINITION

def GET_BREAKS(signal):
    
  y = torch.tensor(signal)

  test = CUSUM_DETECTOR()
  outs = [test.predict_next(y[i]) for i in range(len(y))]

  cps = np.where(list(map(lambda x: x[1], outs)))[0]
  
  return len(cps)


#%% CUSUM DETECTOR

print('\nGetting CUSUM BREAKS ...')

for t in lpc:
    
    print('\n', t, '...')
    
    cq_nn = GET_BREAKS(prices[t])
    cq_da = GET_BREAKS(norm[t]['da'])
    cq_dis = GET_BREAKS(norm[t]['dis'])
    cq_mae = GET_BREAKS(norm[t]['mae'])
    
    df = pd.DataFrame([cq_nn, cq_da, cq_dis, cq_mae]).T
    df.columns = ['PRICE_CUSUM_BREAKS','DA_NORM_PRICE_CUSUM_BREAKS','DIS_NORM_PRICE_CUSUM_BREAKS','MAE_NORM_PRICE_CUSUM_BREAKS']
    
    df.to_excel(outfolder + dpath + t + '.xlsx', index=False)

    if git == 1:
        
        os.system('git add .')
        os.system('git commit -m'+' '+'"cusum breaks - '+dset+' - '+t+'"')
        os.system("git"+" "+"push -u lpc"+" "+git_branch)
        
print('     ...done!')
