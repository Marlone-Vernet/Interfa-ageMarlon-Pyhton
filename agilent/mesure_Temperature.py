# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 10:16:06 2021

@author: admin-si
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 14:12:19 2021

@author: VERNET Marlone

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
get_ipython().run_line_magic('matplotlib', 'qt')

plt.rcParams.update({'font.size': 18})

"""
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["sans-serif"]})"""
plt.rcParams['figure.figsize'] = (6,5)

#%% 

symbol = ["o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d"]

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
Keithley = rm.open_resource('GPIB0::9::INSTR')

Keithley.read_termination = '\n'
Keithley.write_termination = '\n'

Keithley.write('SYSTem:BEEPer:STATe OFF')

Keithley.baud_rate = 57600

N_samples = 360
Buff = 10000
#num_chan = 2
PAUSE = 30
N_channel = 15
plot_chan = [0,1,2,3,4,5,6,7,8,9,11,12,13,14] #PB chan 10

temps = np.zeros((N_samples))
result = dict()
t0 = time.time()

#Keithley.write(":*SRE 1")
# Keithley.write(":TRACe:FEED:CONTrol NEVer")
# Keithley.write(":TRAC:CLE:AUTO ON")
#Keithley.write(":TRAC:FEED SENS")
# Keithley.write(f':TRACe:POINts {Buff}')

#Keithley.write(':TRAC:FEED:CONTrol NEVer')
Keithley.write(':FUNC "RES", (@101:115)')
Keithley.write(':ROUTe:SCAN (@101:115)')
Keithley.write('TRIG:COUNT 1')
Keithley.write('FORM:READ:TIME OFF')

temperature = dict()
for j in range(N_channel):
    temperature[j] = []

data_time = []
t0 = time.time()


fig, axs = plt.subplots()
plt.ion()
plt.grid()
fig.show()

for k in range(N_samples):
    Keithley.write(':READ?')
    t1  = time.time() - t0
    data_time.append(t1)
    
    data_ = Keithley.read_raw()
    string = str(data_)
    list_ = string.split(',')
    long = len(list_)
    for j in range(long):
        chain = list_[j]
        r1 = chain.index('+') 
        r2 = chain.index('E')
        Res = float(chain[r1:r2])*100
        temp = Pt100_temperature( Res )
        temperature[j].append( temp )
            
        
    [axs.plot(data_time, temperature[m], symbol[m]) for m in plot_chan] 

    fig.canvas.draw()
    plt.pause(0.01)    
    
    time.sleep(PAUSE)
    
Keithley.clear()

temperature['t'] = data_time 


with open(folder+name, "wb") as fp:   #Pickling
    pk.dump(temperature, fp)

#%%

plt.figure(2)
plt.plot([temperature[k][-1] for k in range(15)], 'o')

plt.grid()
plt.show()



