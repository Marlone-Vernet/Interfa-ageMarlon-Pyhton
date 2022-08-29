# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 11:33:50 2022

@author: mvern
"""

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

rm = pyvisa.ResourceManager()
rm.list_resources()

print(rm.list_resources())

"""
IF NOThing Happen, then verify that the alim is on USB remote control, to do so: 
    - press 'shift' + 'P-set'
    - press on I/O
    - select USB
    - select TMC (usb common cable)
"""

ITECH = rm.open_resource('USB0::0x2EC7::0x6000::804449011767740005::0::INSTR') #id of the ALIM

ITECH.read_termination = '\r\n'
ITECH.write_termination = '\r\n'

ITECH.write('*RST')
ITECH.write('SYSTem:BEEPer:IMMediate')
ITECH.write('SYSTem:REMote') # lock panel control

ITECH.write('SOURce:FUNCtion VOLTage') # set the alim control on Tension (or Current)
#ITECH.write('SOURce:CURRent:OVER:PROTection:LEVel?')


state = 'on'
ITECH.write(f'OUTPut:STATe {state}') # start the output

ITECH.write('SOURce:VOLTage:SLEW:BOTH 5') #set the time of rise and fall of voltage in sec


volt = [1,5,10]
len_volt = len(volt)
delta_time = 5#3600 #time per plateau in seconds

ITECH.write(f'SOURce:VOLTage:LEVel:IMMediate:AMPLitude {volt[0]}') #set first voltage value of the list
trusty_sleep(delta_time)


#LOOP over the voltage list
for k in range(1,len_volt): 
    ITECH.write(f'SOURce:VOLTage:LEVel:IMMediate:AMPLitude {volt[k]}')
    trusty_sleep(delta_time)
    

ITECH.write('SOURce:VOLTage:SLEW:BOTH 10')
ITECH.write(f'SOURce:VOLTage:LEVel:IMMediate:AMPLitude 0')  #set the time of rise and fall of voltage in sec





# at the end of the run, the alim panel control is unlock
ITECH.write('SYSTem:LOCal')