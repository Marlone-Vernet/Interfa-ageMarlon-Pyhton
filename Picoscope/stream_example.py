# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 17:16:47 2021

@author: admin-si
"""

import ctypes
import numpy as np
from picosdk.ps5000a import ps5000a as ps
import matplotlib.pyplot as plt
from picosdk.functions import adc2mV, assert_pico_ok
import time

import pickle as pk 

# module figures
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
plt.rcParams.update({'font.size': 22})
plt.rcParams["figure.figsize"] = (6,5)
# plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.serif": ["Palatino"]})



def DataToMV(signal, Range, maxADC):
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
    
    t1 = t0 - time.time()
    print(t1)
    print("Conversion to mV DONE")
    
    return signal_ 


#%%


def SpikeFinder(signal):
    """ creates a dict of snapshots """
    
    delta = 10000
    delay = 673
    burstL = 160
    interResp = 3490
    
    TotalSamples = len(signal)
    snapshot = dict()
    index = np.where(signal[:delta] == max(signal[:delta]))
    idx1 = index[0][0] + delay 
    idx2 = index[0][0] + delay + burstL
    snapshot[0] = signal[idx1:idx2]
    
    k = 0
    lastIndex = idx2 
    while lastIndex - delta < TotalSamples:
        k+=1
        index = np.where(signal[ lastIndex:delta ] == max(signal[ lastIndex:delta ]))
        idx1 = lastIndex + index[0][0] + delay 
        idx2 = lastIndex + index[0][0] + delay + burstL
        snapshot[k] = signal[ idx1:idx2 ]
    
    return snapshot 
        
    

    

#%%

"""
numBuffersToCapture = 600
sizeOfOneBuffer = int(2e6)

Works well ! 

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

numBuffersToCapture = 600



# Open PicoScope 5000 Series device
# Resolution set to 12 Bit
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
channel_range = ps.PS5000A_RANGE['PS5000A_50MV']
status["setChA"] = ps.ps5000aSetChannel(chandle,
                                        ps.PS5000A_CHANNEL['PS5000A_CHANNEL_A'],
                                        enabled,
                                        ps.PS5000A_COUPLING['PS5000A_DC'],
                                        channel_range,
                                        analogue_offset)
assert_pico_ok(status["setChA"])


"""################# MODIF #######################

nSegments = 1000
nMaxSamples = ctypes.c_int(500)

status["memory"] = ps.ps5000aMemorySegments(chandle,
                                            nSegments,
                                            ctypes.byref(nMaxSamples))

assert_pico_ok(status["memory"])

"""###############################################

# Size of capture
sizeOfOneBuffer = int(2e6)

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
# adc2mVChAMax = adc2mV(bufferCompleteA, channel_range, maxADC)
DATA = DataToMV(bufferCompleteA, channel_range, maxADC)

T3 = time.time()
print(f"{T3-T2}s")

# Create time data
time_ = np.linspace(0, (totalSamples) * actualSampleIntervalNs, totalSamples)


# Stop the scope
# handle = chandle
status["stop"] = ps.ps5000aStop(chandle)
assert_pico_ok(status["stop"])

# Disconnect the scope
# handle = chandle
status["close"] = ps.ps5000aCloseUnit(chandle)
assert_pico_ok(status["close"])

# Display status returns
print(status)

print(f"Duration of recorded signal: {totalSamples * 48e-9}s")

print("END")


#%%

data = np.ctypeslib.as_array(adc2mVChAMax[:])

folder = '/Users/admin-si/Desktop/faraday/190821/'
name_ = 'test1'
with open(folder+name_, 'wb') as fil:
    pk.dump(data, fil, protocol=pk.HIGHEST_PROTOCOL)

ploot = False 
if ploot:
    plt.figure()
    plt.plot(time_, data)
    plt.xlabel('Time (ns)')
    plt.ylabel('Voltage (mV)')
    plt.show()

#%%

test = DATA[:50000]

xx = 15192
plt.figure()
plt.plot(test[xx+673:xx+5000])
plt.ylabel('Voltage (mV)')
plt.show()


test_result = SpikeFinder(test)