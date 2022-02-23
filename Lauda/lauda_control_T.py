# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 17:44:22 2022

@author: Marlone Vernet

CONTROLLING LAUDA CHILLER 

"""

# COM3,4,5,8 


import pyvisa
import numpy as np
import time as ti

#%%

def trusty_sleep(n):
    start = ti.time()
    while (ti.time() - start < n):
        ti.sleep(n - (ti.time() - start))
    
    end = ti.time() - start
    print(end)

#%%

# parameters 

T_start = 15.0
T_stop = 85.0
T_consigne = np.arange(T_start, T_stop, 5) # in Celcius 

len_T = len(T_consigne)
delta_T = 3600*2.0

rm = pyvisa.ResourceManager()
rm.list_resources()

print( rm.list_resources() )
lauda = rm.open_resource('ASRL9')

lauda.baud_rate = 9600

lauda.read_termination = '\r\n'
lauda.write_termination = '\r\n'


lauda.write('START')
lauda.write('OUT_SP_01_3')
#lauda.write('OUT_SP_00_15.0')


for k in range(len_T):
    print(f'go to{T_consigne[k]}')
    lauda.write(f'OUT_SP_00_{T_consigne[k]}')

    trusty_sleep(delta_T)


lauda.write('OUT_SP_00_20.0')
trusty_sleep(60*10) # temps pour revenir gentiment à 20°C

lauda.write('STOP')

lauda.close()
