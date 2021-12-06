# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 16:29:39 2021

@author: VERNET Marlon 


"""

import sys
sys.path.insert(1,'/Users/admin-si/Desktop/faraday/')

import numpy as np
from module_rapid_block_mode import PHASE_EXTRACT, RAPID_BLOCK_MODE_Faq, BLOCK_MODE_Faq, STREAMING_MODE 
import pickle as pk
import matplotlib.pyplot as plt
import scipy.signal as scs
import time 

# module figures
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
plt.rcParams.update({'font.size': 22})
plt.rcParams["figure.figsize"] = (6,5)
# plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.serif": ["Palatino"]})




#%%
""" BLOCK MODE """ 

# Parameters:
freq = 1000
folder = '/Users/admin-si/Desktop/faraday/261021/' #folder to record the blocks 
dcAmp = 24
freqHZ = 50

nom = f'{dcAmp}dcA_f{freq}Hz_fex{freqHZ}hZ'



x1=0
x2=350

# block mode parameters :
pretrig = 0
postrig = 400
valueThreshold = 40 #en mV !!! il faut tjrs que le threshold < un peu inf à volt_range !!! !!!
volt_range='50MV'

N_block = 6*5000
timebase = 4
time_sampling = 1/freq
delay = 1950 #1950

# Run the acquisition and store the result in a dict
# !!! all the parameters are not here (some are hiden in the module)
print(f'mesure {dcAmp}')
dict_data = RAPID_BLOCK_MODE_Faq(N_block,timebase,time_sampling,pretrig,postrig,volt_range,valueThreshold,delay)


# compute the phase shift 
dphi = np.zeros((N_block,))
s0 = np.array(dict_data[0][x1:x2])
Nm = 100

for k in range(N_block):
    s1 = np.array(dict_data[k][x1:x2])
    dphi[k] = np.nanmean(PHASE_EXTRACT(s0, s1, Nm))


with open(folder+f'snapshot_file_{nom}.pkl','wb') as f:
    pk.dump(dict_data,f)
f.close()

plt.figure()
[plt.plot(dict_data[10+k,:]) for k in range(100)]

plt.show()

##############################################

#plt.figure()
#plt.plot(dict_data[20][:])

#plt.show()


dt = time_sampling
time_ = np.arange(0,N_block,1)*dt


plt.figure()
plt.plot(time_, dphi, '-')
plt.grid()
plt.show()

fac = freq
f, psd = scs.welch(dphi, fs=fac, nperseg=fac)

plt.figure()
plt.loglog(f, psd)

plt.grid()
plt.show()

#%%

Acc = [3.6,4.4,5.4,6.6,7.6,8.6,9.8,10.8,11.8,12.8,14.2,15.4,16.2,
       17.2,18.2,19.2,20.0,21.2,22.2,23.4,24.8,26.8,27.6,28.6] # en mV

Accd = []