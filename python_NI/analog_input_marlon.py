from queue import Queue
from ctypes import byref

import numpy as np

import PyDAQmx

# from ..analog_input import AnalogInput
class AnalogInput(object):
    pass

Task = PyDAQmx.Task if PyDAQmx is not None else object

class DAQmxAnalogInput(AnalogInput):
    def __init__(self, device_info):
        self.device_info = device_info
        #self.name = self.device_info._device_str
        self.name = device_info
        
    def start(self, channel_list, sample_rate, block_size, N_block, volt_range, acquisition_mode):
        self._raw_channel_list = channel_list
        channel_list = ["{}/ai{}".format(self.name, elm) for elm in channel_list]
        print(channel_list)
        self._channel_list = channel_list
        self._task = AITask(channel_list, sample_rate, block_size, volt_range, acquisition_mode)
        self._task.StartTask()

    def get_one_block(self):
        res = self._task.queue.get()
        return {rk:res[k] for rk, k in zip(self._raw_channel_list, self._channel_list)}

    def stop(self):
        self._task.StopTask()


class AITask(Task):
    def __init__(self, channel_list, sample_rate, block_size, volt_range, acquisition_mode=PyDAQmx.DAQmx_Val_RSE):
        Task.__init__(self)
        channel_list_str = ','.join(channel_list)
        n_channel = len(channel_list)
        self.CreateAIVoltageChan(channel_list_str, "", acquisition_mode, -volt_range, volt_range, PyDAQmx.DAQmx_Val_Volts, None)
        self.CfgSampClkTiming("", sample_rate, PyDAQmx.DAQmx_Val_Rising, PyDAQmx.DAQmx_Val_ContSamps, block_size)
        self.AutoRegisterEveryNSamplesEvent(PyDAQmx.DAQmx_Val_Acquired_Into_Buffer, block_size, 0)
        self.AutoRegisterDoneEvent(0)
        self.queue = Queue()
        self.data = np.zeros(n_channel*block_size)
        self.block_size = block_size
        self.n_channel = n_channel
        self.channel_list = channel_list

    def EveryNCallback(self):
        read = PyDAQmx.int32()
        self.ReadAnalogF64(self.block_size, 1, PyDAQmx.DAQmx_Val_GroupByChannel, self.data, self.n_channel*self.block_size, byref(read), None)
        machin = self.data.reshape((self.n_channel,self.block_size))
        res = {}
        for k,name in enumerate(self.channel_list):
            res[name] = machin[k,:].copy()
        self.queue.put(res)
        return 0 # The function should return an integer

    def DoneCallback(self, status):
        return 0 # The function should return an integer







# from ...autodetection.manufacturer import national_instrument

# national_instrument.add_model("analog_input", DAQmxAnalogInput)


#n = self.sampleNumber
#data = np.zeros(n*self.numberOfChannel, dtype=np.float64)
#read = int32()
#try:
#    self.task.ReadAnalogF64(n, 1.0, DAQmx_Val_GroupByChannel, data, n*self.numberOfChannel, byref(read), None)
#except DAQError:
#    logger.info("DAQError in AnalogInput. This try-except is ugly. Remove it if you have a bug to see the error")
#res=dict([(i,None) for i in self.physicalChannel])
#machin = data.reshape((self.numberOfChannel,n))
#for k,name in enumerate(self.physicalChannel):
#    res[name] = machin[k,:]
#res['t'] = np.arange(self.sampleNumber)/float(self.sampleRate)
#return res










