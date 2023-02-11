# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 04:03:04 2023

@author: Andriu
"""


#%% SCRIPT SETUP

import os
import subprocess

import argparse

parser = argparse.ArgumentParser(prog= 'LPC_grid_search',
                                 description = 'Analyse a Time Serie with Rolling LPC model',
                                 epilog = 'Created by Andriu')

parser.add_argument('-ut', '--token', required=True, type=str)
parser.add_argument('-un', '--user', required=True, type=str)
parser.add_argument('-ue', '--email', required=True, type=str)

parser.add_argument('-d','--dset', required=True, type=int, choices=[1,2,3])
parser.add_argument('-s','--slice', required=True, type=int, choices=[1,2,3,4,5,6,7,8,9,10])
parser.add_argument('-g','--grid', required=True, type=str, choices=['short','full'])
parser.add_argument('--git', required=True, type=int, choices=[0,1])

args = parser.parse_args()

# ARGUMENTS
ds = args.dset
sl = args.slice
g = args.grid
git = args.git

slice_str = "slice"+str(sl)
dset_str = "set"+str(ds)
git_branch = dset_str+'/'+slice_str

#%% GITHUB SETUP

print('\nGithub setup ...')

git_url = 'https://github.com/MLPortius/LPC_Finance.git'

# REPO AND USER
git_user = args.user
git_email = args.email
git_token = args.token

# USER ACCESS

print('     ...accesing to repo',git_url)

os.system("git"+" "+"config user.name"+" "+git_user)
os.system("git"+" "+"config user.email"+" "+git_email)
os.system("git"+" "+"config user.password"+" "+git_token)

# REPO SETUP

remotes = subprocess.check_output("git remote")
remotes = remotes.decode().split('\n')[:-1]

if not "lpc" in remotes:
    print('     ...adding lpc remote to repo')
    os.system("git"+" "+"remote add lpc"+" "+git_url)

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

#%% RUN GRID

print('\n')
cmd = 'python'+' '+'grid_search.py'+' '+'-s'+' '+str(sl)+' '+'-d'+' '+str(ds)+' '+'-g'+' '+str(g)+' '+'--git'+' '+str(git)
os.system(cmd)

