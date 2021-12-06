# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 15:16:31 2021

@author: admin-si
"""

import PyDAQmx as nidaq
import numpy as np 
import ctypes as ct


"""This example is a PyDAQmx version of the ContAcq_IntClk.c example
It illustrates the use of callback functions

This example demonstrates how to acquire a continuous amount of
data using the DAQ device's internal clock. It incrementally stores the data
in a Python list.
"""

class CallbackTask:
    def __init__(self):
        running_task = nidaq.Task()
        running_task.__init__(self)
        self.data = np.zeros(1000)
        self.a = []
        self.CreateAIVoltageChan("Dev1/ai0","",nidaq.DAQmx_Val_RSE,-10.0,10.0,nidaq.DAQmx_Val_Volts,None)
        self.CfgSampClkTiming("",10000.0,nidaq.DAQmx_Val_Rising,nidaq.DAQmx_Val_ContSamps,1000)
        self.AutoRegisterEveryNSamplesEvent(nidaq.DAQmx_Val_Acquired_Into_Buffer,1000,0)
        self.AutoRegisterDoneEvent(0)
        
    def EveryNCallback(self):
        int32 = ct.c_long
        read = int32()
        self.ReadAnalogF64(1000,10.0,nidaq.DAQmx_Val_GroupByScanNumber,self.data,1000,nidaq.byref(read),None)
        self.a.extend(self.data.tolist())
        
        # plt.figure(1)            
        # plt.plot(self.data[0])
        
        # plt.grid()
        # plt.show()
        print(self.data[0])
        
        return 0 # The function should return an integer
    
    def DoneCallback(self, status):
        print("Status",status.value)
        return 0 # The function should return an integer


task = CallbackTask()
task.StartTask()

task.EveryNCallback()

task.StopTask()
task.ClearTask()