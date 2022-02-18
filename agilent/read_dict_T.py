# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 09:47:45 2021

@author: VERNET Marlone

"""

import pymeasure as pm
import numpy as np
import time
import pandas as pd
import pickle as pk

from Pt100_resistance_to_temperature import Pt100_temperature

import pyvisa
import matplotlib.pyplot as plt



from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')

plt.rcParams.update({'font.size': 18})

"""
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["sans-serif"]})"""
plt.rcParams['figure.figsize'] = (6,5)

#%%


folder = "C:/Users/admin-si/Desktop/GalliMer/061221/"
name = 'dict_T_15_70.pkl'

with open(folder+name, "rb") as fp:   #Pickling
    dict_T = pk.load(fp)

tt = dict_T['t']

plt.figure(10)
plt.plot(tt, dict_T[0], 'o')
plt.plot(tt, dict_T[1], 's')

plt.grid()
plt.xlabel(r'$t [s]$')
plt.ylabel(r'$T [k]$')
plt.legend()
plt.show()



