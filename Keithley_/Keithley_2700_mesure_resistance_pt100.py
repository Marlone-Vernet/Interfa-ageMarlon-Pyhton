# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 12:03:24 2021

@author: mvernet

"""

import numpy as np
import time
import pyvisa
from Pt100_resistance_to_temperature import Pt100_temperature



def KEITHLEY_ACQUISITION_RESISTANCE(interval_in_ms, number_of_readings):
    """ Parameters :
    interval_in_ms : float; sampling time in ms
    number_of_readings : int, number of measurements
    """

    rm = pyvisa.ResourceManager()
    print( rm.list_resources() )
    Keithley = rm.open_resource( rm.list_resources()[0] ) # set the instrument as the first in the list !
    
    Keithley.read_termination = '\n' # set the termination for read and write
    Keithley.write_termination = '\n'
    
    Keithley.baud_rate = 57600 # set the baud rate
    
         
    Keithley.write(":*RST")
    Keithley.write("::STATus:PRESet")
    Keithley.write(":*CLS")
    Keithley.write("SYSTem:BEEPer:STATe OFF")
    
    
    Keithley.write(":STATus:MEASurement:ENABle 512; *sre 1")
    Keithley.write(":MEASure:RESistance?")
    Keithley.write(":SAMPle:COUNt %d" % number_of_readings)
    Keithley.write(":TRIGger:SOURce BUS")
    Keithley.write(":TRIGger:DELay %f" % (interval_in_ms / 1000.0))
    Keithley.write(":TRACe:POINts %d" % number_of_readings)
    Keithley.write(":TRACe:FEED SENSe1")
    Keithley.write(":TRACe:FEED:CONTrol NEXT")
    
    print('Start acquisition')
    Keithley.write(":INITiate")
    Keithley.assert_trigger()
    Keithley.wait_for_srq()
    
    # result = Keithley.read_raw()
    # result = Keithley.query_ascii_values("TRACe:DATA?") # , converter='f', container = np.array)
    
    result = Keithley.query(":TRACe:DATA?")
    print('acquisition done')
    
    Keithley.write(":TRACe:CLEar")
    Keithley.write(":TRACe:FEED:CONTrol NEXT")
    Keithley.clear()
    
    print('Start data conversion')
    Temps, Temperature = ASCII_query_Pt100(result, number_of_readings)
    
    print('Done')
    return Temps, Temperature





def ASCII_query_Pt100(res_, number_of_readings):
    temperature = np.zeros((number_of_readings,))
    temps = np.zeros((number_of_readings,))
    result_ = '#,'+res_+','
    
    tag_pos = []
    for pos,char in enumerate(result_): # find les '#' which states the end of message sequence
        if char == '#':
            tag_pos.append(pos)
    
    for k in range(number_of_readings):
        """ decrypte les data machines """
        chain = result_[tag_pos[k]+1:tag_pos[k+1]] #find the interval of string = to resistance values
        r1 = chain.index('+') 
        r2 = chain.index('E')
        temperature[k] = Pt100_temperature( float(chain[r1:r2])*1e2 ) # WORKS ONLY FOR Pt100 !!!
        t1 = chain[r2+2:].index('+') #find the interval of string = to time values
        t2 = chain[r2+2:].index('S')
        temps[k] = float(chain[r2+2+t1:r2+2+t2]) # attention : temps en secs, si Taq plus de 9s probleme !!!
        # print(chain)
        
    return temps, temperature


def ASCII_raw_query_Pt100(res_, number_of_readings):
    temperature = np.zeros((number_of_readings,))
    temps = np.zeros((number_of_readings,))
    result_ = '#,'+res_+','
    
    tag_pos = []
    for pos,char in enumerate(result_): # find les '#' which states the end of message sequence
        if char == '#':
            tag_pos.append(pos)
    
    for k in range(number_of_readings):
        """ decrypte les data machines """
        chain = result_[tag_pos[k]+1:tag_pos[k+1]] #find the interval of string = to resistance values
        r1 = chain.index('+') 
        r2 = chain.index('E')
        temperature[k] = Pt100_temperature( float(chain[r1:r2])*1e2 ) # WORKS ONLY FOR Pt100 !!!
        
    return temperature



