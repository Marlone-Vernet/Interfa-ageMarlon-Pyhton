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

folder = "C:/Users/admin-si/Desktop/GalliMer/Nusselt_mesure3/"

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

N_samples = 2 # 32*60*2*2 # N_plateau * N_points/heure * 2 * 2h (14 ???) + 1 pour le 0
Buff = 1000 # buffer size
PAUSE = 5 #30
N_channel = 27 # number of channels
plot_chan = np.arange(0,N_channel,1)  # channels to plot
chani = 1001 # first channel to scan
chanf = 1024 # last channel to scan
RTD = 1026 # RTD measure of thermocouple junction 
chanff = 1027

total = np.arange(1001,1010,1)
#total = [chan_, 1026]

temps = np.zeros((N_samples))
result = dict()
t0 = time.time()


K3706A.write('reset()') # clear buffer memory
K3706A.write(f"buf = dmm.makebuffer({Buff})") # set buffer size

K3706A.write("dmm.func='twowireohms'") # set the type of measurement
K3706A.write("dmm.range=1000") # set the range of ohms (1kOhms)
K3706A.write("dmm.configure.set('myres')") # config the name of the buffer (access name)
K3706A.write(f"dmm.setconfig('{chani}:{chanf}','myres')") # set the channels to scans
K3706A.write(f"dmm.setconfig('{RTD}','myres')") # set the channels to scans

K3706A.write("dmm.func = 'temperature'")
K3706A.write("dmm.transducer = dmm.TEMP_THERMOCOUPLE") # set thermocouple for measurement
K3706A.write("dmm.thermocouple = dmm.THERMOCOUPLE_T") # set type of thermocouple
K3706A.write("dmm.units = dmm.UNITS_CELSIUS") # set units of the thermocouple
K3706A.write("dmm.refjunction = dmm.REF_JUNCTION_INTERNAL") # set type of reference
K3706A.write("dmm.configure.set('mytemp')")
K3706A.write("dmm.setconfig('1025','mytemp')") # set thermocouple channels
K3706A.write("dmm.setconfig('1027','mytemp')") # set thermocouple channels


K3706A.write(f"scan.create('{chani}:{chanff}')") # create the scan
K3706A.write("scan.scancount=1") # number of scan per execution 

temperature = dict() # dictionnary where to store temps data
for j in range(N_channel):
    temperature[j] = []

data_time = []
t0 = time.time()

PLOT = False

if PLOT==True:
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
        if j%2==0 and j!=48 and j!=52:
            temp = Pt100_temperature( float(list_[j]) )
            temperature[j/2].append( temp )
            
        elif j==48:
            temperature[j/2].append( float(list_[j] )) #exception for thermocouple (not a RTD pt100)

        elif j==52:
            temperature[j/2].append( float(list_[j]) )
            
    if PLOT==True:
        [axs.plot(data_time, temperature[m], symbol[m]) for m in plot_chan] 
    
        fig.canvas.draw()
        plt.pause(0.01)    
    
    time.sleep(PAUSE)

K3706A.clear()

tend = time.time() - t0
print(tend)
print(tend//30)

temperature['t'] = data_time 


with open(folder+name, "wb") as fp:   #Pickling
    pk.dump(temperature, fp)

#%%

plt.figure(11)
plt.plot(plot_chan, [temperature[k][-1] for k in range(N_channel)], 'o')

plt.grid()
plt.show()