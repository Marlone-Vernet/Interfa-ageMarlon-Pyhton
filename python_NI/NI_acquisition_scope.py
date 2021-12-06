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

import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as scs

# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib','qt')

%matplotlib qt

plt.rcParams.update({'font.size': 22})

#%%

plt.ion()


sample_rate = 1000
duration = 60 # en s 

channel_list = ['0'] #['1', '2']
block_size = sample_rate//5
N_block = 5*duration
volt_range = 1 # en Volt
acquisition_mode = PyDAQmx.DAQmx_Val_Diff

fig = plt.figure(figsize=(10,7))
ax = fig.subplots(1,2)

fig.show()
fig.canvas.draw()

#task = AnalogInputThreadSimulation('AI')
task = DAQmxAnalogInput('/dev5')
task.start(channel_list, sample_rate, block_size, N_block, volt_range, acquisition_mode)

all_data = np.array([])
for _ in range(N_block):
    data = task.get_one_block()
    all_data = np.concatenate([all_data, data['0']])
    ax[0].clear()
    #ax[0].plot(all_data)
    ax[0].plot(data['0'])
    f,psd = scs.welch(all_data, fs=sample_rate)
    ax[1].clear()
    ax[1].loglog(f,psd)
    fig.canvas.draw()
    plt.pause(0.05)
    
task.stop()