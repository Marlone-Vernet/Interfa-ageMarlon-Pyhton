# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 12:16:58 2021

@author: admin-si
"""

from analog_input_marlon import DAQmxAnalogInput
from time import time
import PyDAQmx
import pickle 

import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as scs

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib','qt')


plt.rcParams.update({'font.size': 22})

#%%

folder="C:/Users/admin-si/Desktop/kepler_couette/160221/"

B0 = 1e-1 #en Tesla
gain = 1000
d = 1e-2 #en metre 

I = [15,30,60,100,200,300,400,500,600,700,800,900,1000]
LI = len(I)
std_t = np.zeros((LI))
std_r = np.zeros((LI))


for k in range(LI):
    number = I [k]
    name1 = f'ut_{number}A1000G.pkl'
    name2 = f'ur_{number}A1000G.pkl'

    with open(folder+name1, "rb") as fp:   #Pickling
        ut = pickle.load(fp)
    with open(folder+name2, "rb") as fp:   #Pickling
        ur = pickle.load(fp)

    ut = ut/(gain*d*B0)
    ur = ur/(gain*d*B0)

    std_t[k] = np.std(ut)
    std_r[k] = np.std(ur)


plt.figure()
plt.plot(I, std_t, 'o-')
plt.semilogx(I, std_r, 's-')
plt.grid()
plt.show()

#%%

folder="C:/Users/admin-si/Desktop/kepler_couette/220221/"

B0 = 0.02 #6e-2 #en Tesla
gain = 1000
input_range = 0.1
d = 1e-2 #en metre 

I = np.array([15,30,60,100,200,300,400,500,600,700,800,900,1000])
LI = len(I)
std_t = np.zeros((LI))
std_r = np.zeros((LI))


for k in range(LI):
    number = I [k]
    # name1 = f'u_{number}A600G.pkl'
    name1 = f'u_{number}A200G.pkl'

    with open(folder+name1, "rb") as fp:   #Pickling
        u = pickle.load(fp)

    ut = u[0]
    ur = u[1]
    ut = ut/(input_range*gain*d*B0)
    ur = ur/(input_range*gain*d*B0)

    std_t[k] = np.std(ut)
    std_r[k] = np.std(ur)

r = 13e-2
rho = 6.44e3
law = np.sqrt(np.array(I)*B0/(4*np.pi*rho*r))

plt.figure()
plt.loglog(I*B0, std_t, 'o-', label=r'$u_{\theta}$')
plt.loglog(I*B0, std_r, 's-', label=r'$u_r$')
plt.loglog(I*B0, law, '--')

plt.xlabel('IB [AT]')
plt.ylabel(r'$\sigma(u)~~[m.s^{-1}]$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

#%%

sig = 3.6e6
mu0 = 4*np.pi*1e-7
h = 1.5e-2
r = 1.3e-3

plt.figure()
plt.loglog(I*B0, std_t, 'o-', label=r'$u_{\theta}$')
plt.loglog(I*B0, std_r, 's-', label=r'$u_r$')
plt.loglog(I*B0, law, '--')

plt.xlabel(r'$ IB [A.T]$')
plt.ylabel(r'$\sigma(u)~~[m.s^{-1}]$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
