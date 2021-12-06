# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 16:18:12 2021

@author: Marlon Vernet

FONCTION : rapid block mode Vernet

##############################################################################

code for rapid block acquisition : 
    
record screen shot of signal (signal frequencyup to ~60MHz (1/16ns))
sampling frequency of the sampling up to ~5kHz

N_block : number of samples (screenshots)
timebase : look at the API -> corresponds to the time scale of 
the recorded signal in the screenshot 4 => 16ns

"""


import pandas as pd
import dask.dataframe as dd


import ctypes
import pickle as pk
import numpy as np

from picosdk.ps5000a import ps5000a as ps


import matplotlib.pyplot as plt
from picosdk.functions import adc2mV, assert_pico_ok, mV2adc
import time
import matplotlib.cm as cm
import matplotlib.colors as colors
import json



def PHASE_EXTRACT(s1,s2, N_m, mode='valid'):
    """
    Extract the phase shift between signals s1 and s2
    
    Parameters
    ----------
    s1 : liste, array, 1D vector
        temporal signal
    s2 : liste, array, 1D vector
        temporal signal.
    N_m : int_ 
        Number of points to do the moving average.
    mode : char. chain
        How the convolution behave on the edges. The default is 'valid'.

    Returns
    -------
    1D array
        Temporal instantaneous phase shift between s1 and s2.
    
    """
    
    s1 = np.asarray(s1)
    s1 = (s1 - np.mean(s1))/np.max(s1)
    s2 = np.asarray(s2)
    s2 = (s2 - np.mean(s2))/np.max(s2)    
    
    s = s1*s2
    mean_s = np.convolve(s, np.ones(N_m)/N_m, mode=mode)#number of points to do the moving average
    
    return np.arccos(2*mean_s)


def DataToMV(signal, Range, maxADC, disp=True):
    """ 
        DataToMV(
                c_short_Array           bufferADC
                int                     range
                c_int32                 maxADC
                )
               
        Takes a buffer of raw adc count values and converts it into millivolts
    """
    t0 = time.time()
    
    channelInputRanges = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
    vRange = channelInputRanges[Range]
    
    signal_ = np.array( signal, dtype=np.float32)
    signal_ = (signal_ * vRange)/maxADC.value 
    
    t1 = time.time() - t0
    if disp:
        print(t1)
        print("Conversion to mV DONE")
    
    return signal_ 

def STREAMING_MODE(numBuffersToCapture = 600,
                   volt_range = 50,
                   sizeOfOneBuffer = int(2e6)):
    
    """
    numBuffersToCapture : number of snapshot to take
    volt_range : range of the recording channel in V, MV...
    sizeOfOneBuffer : number of samples recorded to simultanously send to the computer 
    (to maximize for faster recording...) 
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    """
    
    print("Start")
    print("PARAMETERS:")
    print(f"Number Buffers to capture: {numBuffersToCapture}")  
    print(f"Size of One Buffer: {sizeOfOneBuffer}")
    
    """
    numBuffersToCapture = 600
    sizeOfOneBuffer = int(2e6)
    
    Works well ! for ~1min of acquisition 
    
    """
    
    # set heavy variables to zero (save memory ... )
    bufferAMAX = None
    bufferCompleteA = None 
    DATA = None 
    
    print("Start")
    print("Wait...")
    
    T0 = time.time()
    
    # Create chandle and status ready for use
    chandle = ctypes.c_int16()
    status = {}
    
    # numBuffersToCapture = 600
    
    
    # Open PicoScope 5000 Series device
    # Resolution set to 8, 12 or 16 Bit
    resolution =ps.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_16BIT"]
    # Returns handle to chandle for use in future API functions
    status["openunit"] = ps.ps5000aOpenUnit(ctypes.byref(chandle), None, resolution)
    
    try:
        assert_pico_ok(status["openunit"])
    except: # PicoNotOkError:
    
        powerStatus = status["openunit"]
    
        if powerStatus == 286:
            status["changePowerSource"] = ps.ps5000aChangePowerSource(chandle, powerStatus)
        elif powerStatus == 282:
            status["changePowerSource"] = ps.ps5000aChangePowerSource(chandle, powerStatus)
        else:
            raise
    
        assert_pico_ok(status["changePowerSource"])
    
    
    enabled = 1
    disabled = 0
    analogue_offset = 0.0
    
    
    
    # 10MV,20MV,50MV,100MV,200MV,500MV,1V,2V,5V,10V,20V,50V, PS5000A_MAX_RANGES
    channel_range = ps.PS5000A_RANGE[f'PS5000A_{volt_range}MV']
    status["setChA"] = ps.ps5000aSetChannel(chandle,
                                            ps.PS5000A_CHANNEL['PS5000A_CHANNEL_A'],
                                            enabled,
                                            ps.PS5000A_COUPLING['PS5000A_DC'],
                                            channel_range,
                                            analogue_offset)
    assert_pico_ok(status["setChA"])
    
    
    # Size of capture
    # sizeOfOneBuffer = int(2e6)
    
    totalSamples = sizeOfOneBuffer * numBuffersToCapture
    
    # Create buffers ready for assigning pointers for data collection
    bufferAMax = np.zeros(shape=sizeOfOneBuffer, dtype=np.int16)
    
    #memory_segment = np.arange(0,nSegments,1)
    memory_segment = 0
    
    # Set data buffer location for data collection from channel A
    # handle = chandle
    # source = PS5000A_CHANNEL_A = 0
    # pointer to buffer max = ctypes.byref(bufferAMax)
    # pointer to buffer min = ctypes.byref(bufferAMin)
    # buffer length = maxSamples
    # segment index = 0
    # ratio mode = PS5000A_RATIO_MODE_NONE = 0
    status["setDataBuffersA"] = ps.ps5000aSetDataBuffers(chandle,
                                                         ps.PS5000A_CHANNEL['PS5000A_CHANNEL_A'],
                                                         bufferAMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                                                         None,
                                                         sizeOfOneBuffer,
                                                         memory_segment,
                                                         ps.PS5000A_RATIO_MODE['PS5000A_RATIO_MODE_NONE'])
    assert_pico_ok(status["setDataBuffersA"])
    
    
    
    # Begin streaming mode:
    sampleInterval = ctypes.c_int32(40)
    sampleUnits = ps.PS5000A_TIME_UNITS['PS5000A_NS']
    # We are not triggering:
    maxPreTriggerSamples = 0
    autoStopOn = 1
    # No downsampling:
    downsampleRatio = 1
    status["runStreaming"] = ps.ps5000aRunStreaming(chandle,
                                                    ctypes.byref(sampleInterval),
                                                    sampleUnits,
                                                    maxPreTriggerSamples,
                                                    totalSamples,
                                                    autoStopOn,
                                                    downsampleRatio,
                                                    ps.PS5000A_RATIO_MODE['PS5000A_RATIO_MODE_NONE'],
                                                    sizeOfOneBuffer)
    assert_pico_ok(status["runStreaming"])
    
    actualSampleInterval = sampleInterval.value
    actualSampleIntervalNs = actualSampleInterval 
    
    print("Capturing at sample interval %s ns" % actualSampleIntervalNs)
    print(f"Corresponding acquisition frequency: {(1/(actualSampleIntervalNs*1e-9))/1e6}MHz")
    
    # We need a big buffer, not registered with the driver, to keep our complete capture in.
    bufferCompleteA = np.zeros(shape=totalSamples, dtype=np.int16)
    global nextSample, autoStopOuter, wasCalledBack
    nextSample = 0
    autoStopOuter = False
    wasCalledBack = False
    
    def streaming_callback(handle, noOfSamples, startIndex, overflow, triggerAt, triggered, autoStop, param):
        global nextSample, autoStopOuter, wasCalledBack
        wasCalledBack = True
        destEnd = nextSample + noOfSamples
        sourceEnd = startIndex + noOfSamples
        bufferCompleteA[nextSample:destEnd] = bufferAMax[startIndex:sourceEnd]
        nextSample += noOfSamples
        if autoStop:
            autoStopOuter = True
        
        return True 
    
    
    # Convert the python function into a C function pointer.
    cFuncPtr = ps.StreamingReadyType(streaming_callback)
    
    # Fetch data from the driver in a loop, copying it out of the registered buffers and into our complete one.
    while nextSample < totalSamples and not autoStopOuter:
        wasCalledBack = False
        status["getStreamingLastestValues"] = ps.ps5000aGetStreamingLatestValues(chandle, cFuncPtr, None)
        if not wasCalledBack:
            # If we weren't called back by the driver, this means no data is ready. Sleep for a short while before trying
            # again.
            time.sleep(0.001)
    
    print("Done grabbing values.")
    T1 = time.time()
    print(f"{T1-T0}s")
    
    # Find maximum ADC count value
    # handle = chandle
    # pointer to value = ctypes.byref(maxADC)
    maxADC = ctypes.c_int16()
    status["maximumValue"] = ps.ps5000aMaximumValue(chandle, ctypes.byref(maxADC))
    assert_pico_ok(status["maximumValue"])
    
    T2 = time.time()
    print(f"{T2-T1}s")
    
    # Convert ADC counts data to mV
    DATA = DataToMV(bufferCompleteA, channel_range, maxADC)
    
    T3 = time.time()
    print(f"{T3-T2}s")
    
    status["stop"] = ps.ps5000aStop(chandle)
    assert_pico_ok(status["stop"])
    
    status["close"] = ps.ps5000aCloseUnit(chandle)
    assert_pico_ok(status["close"])
    
    print(status)
    
    print(f"Duration of recorded signal: {totalSamples * 48e-9}s")
    
    print("END")

    return DATA 


def BLOCK_MODE_Faq(N_block,
                   timebase,
                   time_sampling,
                   preTriggerSamples = 0, 
                   postTriggerSamples = 600, 
                   volt_range='50MV',
                   valueThreshold=50):
    """
    source = Channel A
    resolution 16Bits (other values 8 & 12)
    N_block : Number of block
    timebase : time interval depends on 8bit or 12bit or 16bit ... see pdf guide (ex: 4 in 16bits -> 16ns)
    time_pause_between_block = 0.0002 # must be > or = 0.0001 == sampling time
    preTriggerSamples : number of points to record before trigger
    postTriggerSamples : number of points to record after trigger (depends on the flight time)
    volt_range # Possible values: 10MV,20MV,50MV,100MV,200MV,500MV,1V,2V,5V,10V,20V,50V, PS5000A_MAX_RANGES
    valueThreshold : int, threshold in mV
        
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    """
    print("Start")
    print("PARAMETERS:")
    print(f"sampling frequency : {1/time_sampling}Hz")

    delay = 2000 #delay before the record in Nbre of samples !!! If too short, prog crash !
    time_pause_between_block = time_sampling
    maxSamples = preTriggerSamples + postTriggerSamples
    # Create chandle and status ready for use
    chandle = ctypes.c_int16()
    status = {}
    
    # serial = 'HT447/0154'
    # serial_number = ctypes.c_char_p(bytes(serial, 'utf-8')) #Or try just None instead 
    
    
    # Open 5000 series PicoScope
    # Resolution set to ? Bit
    resolution = ps.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_16BIT"]
    # Returns handle to chandle for use in future API functions
    status["openunit"] = ps.ps5000aOpenUnit(ctypes.byref(chandle), None, resolution)
    
    try:
        assert_pico_ok(status["openunit"])
    except: # PicoNotOkError:
    
        powerStatus = status["openunit"]
    
        if powerStatus == 286:
            status["changePowerSource"] = ps.ps5000aChangePowerSource(chandle, powerStatus)
        elif powerStatus == 282:
            status["changePowerSource"] = ps.ps5000aChangePowerSource(chandle, powerStatus)
        else:
            raise
    
        assert_pico_ok(status["changePowerSource"])    
    
    
    channel = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
    # enabled = 1
    coupling_type = ps.PS5000A_COUPLING["PS5000A_DC"]
    chARange = ps.PS5000A_RANGE[f"PS5000A_{volt_range}"] # 10MV,20MV,50MV,100MV,200MV,500MV,1V,2V,5V,10V,20V,50V, PS5000A_MAX_RANGES
    # analogue offset = 0 V
    status["setChA"] = ps.ps5000aSetChannel(chandle, channel, 1, coupling_type, chARange, 0)
    assert_pico_ok(status["setChA"])

    timeIntervalns = ctypes.c_float()
    returnedMaxSamples = ctypes.c_int32()
    status["getTimebase2"] = ps.ps5000aGetTimebase2(chandle, timebase, maxSamples, ctypes.byref(timeIntervalns), ctypes.byref(returnedMaxSamples), 0)
    assert_pico_ok(status["getTimebase2"])
    
    maxADC = ctypes.c_int16()
    status["maximumValue"] = ps.ps5000aMaximumValue(chandle, ctypes.byref(maxADC))
    assert_pico_ok(status["maximumValue"])
    
    source = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
    threshold = int(mV2adc(valueThreshold,chARange, maxADC)) #Threshold en mV ici
    
    status["trigger"] = ps.ps5000aSetSimpleTrigger(chandle, 1, source, threshold, 2, 0, 1000) #1000 : autotrigger in ms
    assert_pico_ok(status["trigger"])
    
    t0 = time.time()
    
    status["triggerDelay"] = ps.ps5000aSetTriggerDelay(chandle,delay)

    data = {}
    t1 = time.time()
    for k in range(N_block):
        
        status["runBlock"] = ps.ps5000aRunBlock(chandle, preTriggerSamples, postTriggerSamples, timebase, None, 0, None, None)
        assert_pico_ok(status["runBlock"])
        
        # status["Ready"] = ps.ps5000aBlockReady(chandle, status, None)
        # assert_pico_ok(status["Ready"])
        
        bufferAMax_ = (ctypes.c_int16 * maxSamples)()
        bufferAMin_ = (ctypes.c_int16 * maxSamples)()
        status["setDataBuffer"] = ps.ps5000aSetDataBuffers(chandle, source, ctypes.byref(bufferAMax_), ctypes.byref(bufferAMin_), maxSamples,0, 0)
        assert_pico_ok(status["setDataBuffer"])
        
        time.sleep( time_pause_between_block )
        
        cmaxSamples = ctypes.c_int32(maxSamples)
        downSample = ps.PS5000A_RATIO_MODE["PS5000A_RATIO_MODE_NONE"]
        overflow = ctypes.c_int16()
        status["GetValues"] = ps.ps5000aGetValues(chandle, 0, ctypes.byref(cmaxSamples), 0, 0, 0, ctypes.byref(overflow))

        data_ = np.ctypeslib.as_array(adc2mV(bufferAMax_, chARange, maxADC))
        data[k] = np.array(data_,dtype=np.float32)    

    t2 = time.time() - t1 #quite long: due to extraction of the data from pico to PC
    print(f"Duration time : {t2}")
    print(f"Effective frequency : {N_block/t2}")
    time.sleep(1)
    print('Record ok')
    
    
    status["stop"] = ps.ps5000aStop(chandle)
    assert_pico_ok(status["stop"])
    
    status["close"]=ps.ps5000aCloseUnit(chandle)
    assert_pico_ok(status["close"])
    
    print(status)
    
    print("End ok")
    tout = time.time() - t0
    print(f"total time = {tout}")
    print("Finished")
    
    return data # 1st key : time coordinate, 2nd key : coordinate in the snapshot

        

def RAPID_BLOCK_MODE_Faq(N_block, 
                         timebase, 
                         time_sampling, 
                         preTriggerSamples = 0, 
                         postTriggerSamples = 600, 
                         volt_range='50MV',
                         valueThreshold=50,
                         delay=1950):
    """
    source = Channel A
    resolution 16Bits (other values 8 & 12)
    N_block : Number of block
    timebase : time interval depends on 8bit or 12bit or 16bit ... see pdf guide (ex: 4 in 16bits -> 16ns)
    time_pause_between_block = 0.0002 # must be > or = 0.0001 == sampling time
    preTriggerSamples : number of points to record before trigger
    postTriggerSamples : number of points to record after trigger (depends on the flight time)
    volt_range # Possible values: 10MV,20MV,50MV,100MV,200MV,500MV,1V,2V,5V,10V,20V,50V, PS5000A_MAX_RANGES
    valueThreshold : int, threshold in mV
    delay: int, number of sample before to trigger the recording  (If too short, prog crash !..)
    
    ATTENTION : rapid_block_mode record always at ~burst frequency (?) !!!
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    """
    print("Start")
    print("PARAMETERS:")
    print(f"delay = 1950 samples")

    time_pause_between_block = time_sampling
    maxSamples = preTriggerSamples + postTriggerSamples
    # Create chandle and status ready for use
    chandle = ctypes.c_int16()
    status = {}
    
    # serial = 'HT447/0154'
    # serial_number = ctypes.c_char_p(bytes(serial, 'utf-8')) #Or try just None instead 
    
    
    # Open 5000 series PicoScope
    # Resolution set to ? Bit
    resolution = ps.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_16BIT"]
    # Returns handle to chandle for use in future API functions
    status["openunit"] = ps.ps5000aOpenUnit(ctypes.byref(chandle), None, resolution)
    
    try:
        assert_pico_ok(status["openunit"])
    except: # PicoNotOkError:
    
        powerStatus = status["openunit"]
    
        if powerStatus == 286:
            status["changePowerSource"] = ps.ps5000aChangePowerSource(chandle, powerStatus)
        elif powerStatus == 282:
            status["changePowerSource"] = ps.ps5000aChangePowerSource(chandle, powerStatus)
        else:
            raise
    
        assert_pico_ok(status["changePowerSource"])    
    
    
    channel = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
    # enabled = 1
    coupling_type = ps.PS5000A_COUPLING["PS5000A_DC"]
    chARange = ps.PS5000A_RANGE[f"PS5000A_{volt_range}"] # 10MV,20MV,50MV,100MV,200MV,500MV,1V,2V,5V,10V,20V,50V, PS5000A_MAX_RANGES
    # analogue offset = 0 V
    status["setChA"] = ps.ps5000aSetChannel(chandle, channel, 1, coupling_type, chARange, 0)
    assert_pico_ok(status["setChA"])
    
    maxADC = ctypes.c_int16()
    status["maximumValue"] = ps.ps5000aMaximumValue(chandle, ctypes.byref(maxADC))
    assert_pico_ok(status["maximumValue"])
    
    source = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
    threshold = int(mV2adc(valueThreshold,chARange, maxADC)) #Threshold en mV ici
    
    status["trigger"] = ps.ps5000aSetSimpleTrigger(chandle, 1, source, threshold, 2, 0, 1000) #1000 : autotrigger in ms
    assert_pico_ok(status["trigger"])
    
    t0 = time.time()
    
    status["triggerDelay"] = ps.ps5000aSetTriggerDelay(chandle,delay)
    
    timeIntervalns = ctypes.c_float()
    returnedMaxSamples = ctypes.c_int32()
    status["getTimebase2"] = ps.ps5000aGetTimebase2(chandle, timebase, maxSamples, ctypes.byref(timeIntervalns), ctypes.byref(returnedMaxSamples), 0)
    assert_pico_ok(status["getTimebase2"])
    
    overflow = ctypes.c_int16()
    cmaxSamples = ctypes.c_int32(maxSamples)
    
    status["MemorySegments"] = ps.ps5000aMemorySegments(chandle, N_block, ctypes.byref(cmaxSamples))
    assert_pico_ok(status["MemorySegments"])
    status["SetNoOfCaptures"] = ps.ps5000aSetNoOfCaptures(chandle, N_block)
    assert_pico_ok(status["SetNoOfCaptures"])
    
    status["runBlock"] = ps.ps5000aRunBlock(chandle, preTriggerSamples, postTriggerSamples, timebase, None, 0, None, None)
    assert_pico_ok(status["runBlock"])
    
    
    source = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
    
    # set pointer of buffer ?
    # genere la liste de block à évacuer du scope 
    bufferAMax_ = ((ctypes.c_int16 * maxSamples)*N_block)()
    bufferAMin_ = ((ctypes.c_int16 * maxSamples)*N_block)()
    print("Running...")
    # LOOP of rapid screenshot acquisition
    t1 = time.time()
    for k in range(N_block):
        # t_=time.time()
        status["setDataBuffersA"] = ps.ps5000aSetDataBuffers(chandle, source, ctypes.byref(bufferAMax_[k]), ctypes.byref(bufferAMin_[k]), maxSamples,k, 0)
        assert_pico_ok(status["setDataBuffersA"])
        # time.sleep( time_pause_between_block )
    
    t2 = time.time() - t1 #quite long: due to extraction of the data from pico to PC
    
    print(f"Time in the block loop = {t2}")
    #print(f"Effective frequency = {N_block/t2}")
    print("Block loop ok")
        
    
    overflow = (ctypes.c_int16 * N_block)()
    cmaxSamples = ctypes.c_int32(maxSamples)
    
    ready = ctypes.c_int16(0)
    check = ctypes.c_int16(0)
    while ready.value == check.value:
        status["isReady"] = ps.ps5000aIsReady(chandle, ctypes.byref(ready))
    
    status["GetValuesBulk"] = ps.ps5000aGetValuesBulk(chandle, ctypes.byref(cmaxSamples), 0, N_block-1, 0, 0, ctypes.byref(overflow))
    assert_pico_ok(status["GetValuesBulk"])
    
    time.sleep(0.1)
    print('Get values ok')
    
    
    # generate the dict containing all the data
    data = np.zeros((N_block,maxSamples))
    
    # withdraw the data for each block
    for k in range(N_block):
        data_ = DataToMV(bufferAMax_[k], chARange, maxADC, False)
        data[k,:] = np.array(data_,dtype=np.float32)

    
    time.sleep(0.1)
    print('Record ok')
    
    
    status["stop"] = ps.ps5000aStop(chandle)
    assert_pico_ok(status["stop"])
    
    status["close"]=ps.ps5000aCloseUnit(chandle)
    assert_pico_ok(status["close"])
    
    print(status)
    
    print("End ok")
    tout = time.time() - t0
    print(f"total time = {tout}")
    print("Finished")
    
    return data # 1st key : time coordinate, 2nd key : coordinate in the snapshot
