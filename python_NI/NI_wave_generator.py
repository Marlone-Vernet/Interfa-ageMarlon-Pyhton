# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:20:51 2020

NI output generator 

@author: VERNET
"""


import PyDAQmx as nidaq
import numpy as np
import ctypes as ct
import matplotlib.pyplot as plt
import time
import scipy.signal
import pickle 
import matplotlib.animation as animation

%matplotlib qt


#%%
int32 = ct.c_long
read = int32()
fw = 10# frequency of the sine in Hz
amp = 1# amp 0 to max in volt 
offset = 0
sample_rate = int(1e3)

analog_output = nidaq.Task()
analog_output.CreateAOFuncGenChan("Dev2/ai1","",nidaq.DAQmx_Val_Sine,fw,amp,offset)
analog_output.CfgSampClkTiming(0,sample_rate,nidaq.DAQmx_Val_Falling,nidaq.DAQmx_Val_ContSamps,100)

analog_output.wait_until_done()
analog_output.stop()
analog_output.close()


