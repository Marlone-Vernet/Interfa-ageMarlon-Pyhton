# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 11:22:31 2021

@author: VERNET Marlone

"""


#from tpmontrouge.instrument.analog_input.test.simu_ai import AnalogInputThreadSimulation
#from tpmontrouge.instrument.daqmx.analog_input import DAQmxAnalogInput

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

folder="C:/Users/admin-si/Desktop/Faraday_2_ML/ ????? /"

RECORDING = True
number = 1
freq_ = 30

# name = f'low_noise_ut_{number}_A1000G.pkl'
name1 = f'farad_{number}f{freq_}Hz.pkl'
# name2 = f'ur_{number}A1000G.pkl'

t1 = time()

plt.ion()

""" NI parameter """
DEV = '/dev2'
sample_rate = 2000
duration = 120 # en s 

channel_list = ['0'] #['0', '1', '2'] #['1', '2'] #list of the recorded channels
N_ch = len(channel_list)
block_size = sample_rate//5
N_block = 5*duration
volt_range = 1 # en Volt 

""" welch method parameters """
signal_size = sample_rate*duration
n = int(np.log(signal_size)/np.log(2)) - 1
segment_size = 2**n
number_segment = int(2*signal_size/segment_size) - 1
overlap_size = int( (signal_size - segment_size)/(number_segment - 1) )

"""mode diff, single_ended"""
acquisition_mode = PyDAQmx.DAQmx_Val_Diff
# acquisition_mode = PyDAQmx.DAQmx_Val_PseudoDiff
# acquisition_mode = PyDAQmx.DAQmx_Val_RSE


fig = plt.figure(figsize=(15,15))
ax = fig.subplots(3,N_ch)

fig.show()
fig.canvas.draw()

#task = AnalogInputThreadSimulation('AI')
task = DAQmxAnalogInput(DEV)
task.start(channel_list, sample_rate, block_size, N_block, volt_range, acquisition_mode)
t0 = time()
t=0

all_in = dict() #data are recorded in a dict, each key == one channel

for i in range(N_ch):
    all_in[i] = np.array([])

for _ in range(N_block):
    t=time() - t0
    # t+=block_size/sample_rate
    data = task.get_one_block()
    for k in range(N_ch):
        all_in[k] = np.concatenate([all_in[k], data[ channel_list[k] ]])
        ax[0,k].clear()
        #ax[0].plot(all_1)
        ax[0,k].plot(data[ channel_list[k] ])
        ax[0,k].set_title('t='+str(round(t,2)))
        
        ax[1,k].clear()
        #ax[0].plot(all_1)
        ax[1,k].plot(all_in[ k ])
        # ax[1,k].set_title('t='+str(round(t,2)))
        
        if len(all_in[k]) > segment_size:
            signal = all_in[k] - np.mean(all_in[k])
            f,psd = scs.welch(signal, fs=sample_rate, nperseg=segment_size, noverlap=overlap_size)
            ax[2,k].clear()
            ax[2,k].loglog(f,psd)

    fig.canvas.draw()
    plt.pause(0.01)
task.stop()

if RECORDING == True:

    with open(folder+name1, "wb") as fp:   #Pickling
        pickle.dump(all_in, fp)

print(time() - t1)


#%%
number = 100
name = f'low_noise_ut_{number}_A1000G.pkl'


with open(folder+name_, "rb") as fp:   #Pickling
    M = pickle.load(fp)


plt.figure()
plt.plot(M)
plt.grid()
plt.show()