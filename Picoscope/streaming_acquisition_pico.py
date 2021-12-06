# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 11:45:16 2021

@author: VERNET Marlone
"""


import sys
sys.path.insert(1,'/Users/admin-si/Desktop/faraday/')

import numpy as np
from module_rapid_block_mode import PHASE_EXTRACT, STREAMING_MODE 
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
""" STREAMING MODE """

# Parameters:
freq = 500
folder = '/Users/admin-si/Desktop/faraday/210921/' #folder to record the blocks 
dcAmp = 'test'
freq_ = 30
nom = f'{dcAmp}dcA_f{freq_}Hz'



freqHZ = 30

x1=0
x2=350

# block mode parameters :
"""
pretrig = 0
postrig = 600
valueThreshold = 50 #en mV
volt_range='200MV'

N_block = 100
timebase = 4
time_sampling = 1/freq
"""
# Run the acquisition and store the result in a dict
# !!! all the parameters are not here (some are hiden in the module)
# dict_data = RAPID_BLOCK_MODE_Faq(N_block,timebase,time_sampling,pretrig,postrig,volt_range='200MV')
# dict_data = BLOCK_MODE_Faq(N_block,timebase,time_sampling,pretrig,postrig,volt_range='200MV')

data = STREAMING_MODE(10) # normal value numBuffer = 600 !!! à 1000 PROBLEME...!

with open(folder+f'continuous__{freqHZ}_{nom}.pkl','wb') as f:
    pk.dump(data,f)
f.close()

plt.figure()
plt.plot(data[:10000])
#plt.plot(data)

plt.grid()
plt.show()


#%%


BurstTime = 200e-6 # 200µs
SamplingTime = 48e-9 # 48ns
FreqBurst = 1/BurstTime

subset = data[:10000]
idx = np.where(subset == max(subset))

x0 = idx[0][0]
delay = 650
interB = int(BurstTime/SamplingTime) #7694 - (delay + x0) 
Blargeur = 160
long = len(data) - (x0+delay)

N_block = long//interB 

dphi = np.zeros((N_block,))
s0 = np.array(data[ x0 + delay:x0 + delay + Blargeur])
Nm = 100

t0 = time.time()
for k in range(N_block):
    subset = data[k*interB:k*interB + 10000]
    idx = np.where(subset == max(subset))
    x0 = idx[0][0]
    x1 = x0 + delay 
    x2 = x0 + delay + Blargeur 
    s1 = np.array(subset[x1:x2])
    dphi[k] = np.nanmean(PHASE_EXTRACT(s0, s1, Nm))
    
    # plt.figure(1)
    # plt.plot(s1)
    # plt.show()
t1 = t0 - time.time()
print(f"Post traitment duration : {t1}s")

m_ = 100
post_trait = np.convolve(dphi, np.ones(m_)/m_, mode='valid')
plt.figure()
plt.plot(post_trait)

plt.grid()
plt.show()

f, psd = scs.welch(post_trait, fs=FreqBurst, nperseg=FreqBurst)

plt.figure()
plt.loglog(f, psd)

plt.grid()
plt.show()