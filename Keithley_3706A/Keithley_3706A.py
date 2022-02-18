# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 14:46:53 2022

@author: Marlone Vernet
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
get_ipython().run_line_magic('matplotlib','qt')

plt.rcParams.update({'font.size': 18})

#%%




symbol = ["o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d","o","v","^","<",">","1","2"]

folder = "C:/Users/admin-si/Desktop/GalliMer/tests_15T/020222/"

print("Enter T interieur")
Tin = input()
print("Enter T exterieur")
Text = input()

name = f'dict_T_{Text}_{Tin}.pkl'
#name = f'dict_T_30_60.pkl'


rm = pyvisa.ResourceManager()
rm.list_resources()

print( rm.list_resources() )
K3706A = rm.open_resource('GPIB0::16::INSTR')

K3706A.read_termination = '\n'
K3706A.write_termination = '\n'

#K3706A.write('SYSTem:BEEPer:STATe OFF')

#K3706A.baud_rate = 57600

N_samples = 3
Buff = 1000 # buffer size
PAUSE = 5
N_channel = 24 # number of channels
plot_chan = np.arange(0,N_channel,1)  # channels to plot
chani = 1001 # first channel to scan
chanf = 1024 # last channel to scan

temps = np.zeros((N_samples))
result = dict()
t0 = time.time()


K3706A.write('reset()') # clear buffer memory
K3706A.write(f"buf = dmm.makebuffer({Buff})") # set buffer size

K3706A.write("dmm.func='twowireohms'") # set the type of measurement
K3706A.write("dmm.range=1000") # set the range of ohms (1kOhms)
K3706A.write("dmm.configure.set('myres')") # config the name of the buffer (access name)
K3706A.write(f"dmm.setconfig('{chani}:{chanf}','myres')") # set the channels to scans
K3706A.write(f"scan.create('{chani}:{chanf}')") # create the scan
K3706A.write("scan.scancount=1") # number of scan per execution 


temperature = dict() # dictionnary where to store temps data
for j in range(N_channel):
    temperature[j] = []

data_time = []
t0 = time.time()


fig, axs = plt.subplots()
plt.ion()
plt.grid()
fig.show()

for k in range(N_samples):
    K3706A.write('scan.execute(buf)') # execute the scan
    K3706A.write(f'printbuffer(1,{N_channel},buf,buf.relativetimestamps)') # withdraw the data from the buffer
    
    t1  = time.time() - t0
    data_time.append(t1)
    
    data_ = K3706A.read_raw() # access the data from the buffer 
    # the following lines extract the values of temp from strings
    string = str(data_)
    list_ = string.split(',')
    long = len(list_)
    temp = Pt100_temperature( float(list_[0][2:]) )
    temperature[0].append( temp )
    for j in range(2,long):
        if j%2==0:
            temp = Pt100_temperature( float(list_[j]) )
            temperature[j/2].append( temp )
            

    [axs.plot(data_time, temperature[m], symbol[m]) for m in plot_chan] 

    fig.canvas.draw()
    plt.pause(0.01)    
    
    time.sleep(PAUSE)

K3706A.clear()


temperature['t'] = data_time 


with open(folder+name, "wb") as fp:   #Pickling
    pk.dump(temperature, fp)
