# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 15:34:50 2021

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
folder = '/Users/admin-si/Desktop/faraday/200921/'
folder2 = '/Users/admin-si/Desktop/faraday/200921/'


#dcA = [0,1,2,3,4,5,6,7,8,9]
#dcA = [0,10,20,30,40,41,42,43,44,45,46,48,49,50,51,52,53,54,55,56,57,58,59,60]

ab = [0,100,200,300]

dc = np.arange(400,502,2)

dcA = np.concatenate((ab,dc))

LdcA = len(dcA)
freqHZ = 30
PLOT = False

for meas in range(LdcA):
    
    dcAmp = dcA[meas]
    freq_ = 30
    nom = f'{dcAmp}dcA_f{freq_}Hz'
    
    with open(folder+f'continuous__{freqHZ}_{nom}.pkl','rb') as F:
        data = pk.load(F)
    
    
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
    t1 = time.time() - t0
    print(f"Post traitment duration : {t1}s")
    
    m_ = 100
    post_trait = np.convolve(dphi, np.ones(m_)/m_, mode='valid')
    
    if PLOT:
        plt.figure(meas)
        plt.plot(post_trait, 'o')
        
        plt.grid()
        plt.show()
    
    with open(folder2+f'POST_trait___{nom}.pkl','wb') as f:
        pk.dump(post_trait,f)
        f.close()
    
    """
    f, psd = scs.welch(post_trait, fs=FreqBurst, nperseg=FreqBurst)
    
    plt.figure()
    plt.loglog(f, psd)
    
    plt.grid()
    plt.show()"""