# -*- coding: utf-8 -*-
"""
Created on Tue May 31 19:01:15 2022

@author: Marlone Vernet

"""

import PyDAQmx as ni
import numpy as np
import ctypes as ct
import time as ti 

#%%

def trusty_sleep(n):
    start = ti.time()
    while (ti.time() - start < n):
        ti.sleep(n - (ti.time() - start))
    
    end = ti.time() - start
    print(end)

#%%

""" Lire la doc SGA 40/375 pour les branchements """

dev = 'dev7'


task = ni.Task()
task.CreateAOVoltageChan(f"/{dev}/ao0","",-10.0,10.0,ni.DAQmx_Val_Volts,None)
task.CreateAOVoltageChan(f"/{dev}/ao1","",-10.0,10.0,ni.DAQmx_Val_Volts,None)

task.StartTask()


value1 = 0 #4.615 + (k - 15)*0.0145  ## 9.6e-3 V/G
value2 = 0#1.835 + (k - 5)*7.4e-3 #1.591 + N*7.44e-3 ## 7.4e-3 V/G

value = np.array((value1,value2), dtype=np.float64)

# task.WriteAnalogScalarF64(2,10.0,value,None) #for one channel output
nb_samps_per_chan = 1
int32 = ct.c_long
written = int32()

task.WriteAnalogF64(
            nb_samps_per_chan, #number of channel
            0,
            10.0,
            ni.DAQmx_Val_GroupByChannel,
            value.ravel(),
            ni.byref(written),
            None) #for several channel output



n_mesure = 15
delta_T = 30*60


for k in range(n_mesure):
    value1 = 4.615 + (k - 15)*0.0145  ## 9.6e-3 V/G
    value2 = 1.835 + (k - 5)*7.4e-3 #1.591 + N*7.44e-3 ## 7.4e-3 V/G
    
    value = np.array((value1,value2), dtype=np.float64)
    
    # task.WriteAnalogScalarF64(2,10.0,value,None) #for one channel output
    nb_samps_per_chan = 1
    int32 = ct.c_long
    written = int32()
    
    task.WriteAnalogF64(
                nb_samps_per_chan, #number of channel
                0,
                10.0,
                ni.DAQmx_Val_GroupByChannel,
                value.ravel(),
                ni.byref(written),
                None) #for several channel output
    
    trusty_sleep(delta_T)


task.StopTask()


