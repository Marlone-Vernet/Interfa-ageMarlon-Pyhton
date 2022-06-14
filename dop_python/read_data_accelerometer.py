# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:47:05 2020

Read data from accelerometer

@author: Vernet
"""
import DOPpy as dop

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pickle 

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib','qt')

plt.rcParams.update({'font.size': 22})

#%%
folder="C:/Users/Gissinger/Desktop/faraday/"
current=[0.1,0.3,0.7,0.9,1.2,1.6,1.7,2.0,2.3,2.7]
I = ['01','03','07','09','12','16','17','20','23','27']

amp_f=[]
amp_omega=[]
amp_acc=[]

for i in range(1,5):

    data_ = dop.DOP('test'+str(i)+'.BDD')
    channel=1
    TBP=4.6*1e-3
    #TBP=26.8*1e-3
    dx=0.228

    echo = data_.getEcho(channel)
    N_time,N_gate=np.shape(echo)
    time_ = TBP*np.arange(0,len(echo),1)

    offset=15 #first gate position in mm
    
    Max_list = []
    for k in range(N_time):
        idx = np.where(echo[k,:] == max(echo[k,:]))
        Max_list.append(idx[0][0])
    
    f,psd=scipy.signal.welch(Max_list,fs=1/TBP,nperseg=250)

    amp_f.append(3*np.std(Max_list))
    
    amp_omega.append(psd[23])
    
    #accelerometer part GET THE AMPLITUDE
    name="accelero_"+str(i)+"A.txt"
    
    with open(folder+name, "rb") as fp:   #Pickling
        M = pickle.load(fp)
    
    gain=31.6
    amp_acc.append(gain*3*np.std(M[1]))

plt.figure()
plt.plot(current,amp_acc, '-o')

plt.grid()
plt.show()


plt.figure()
plt.plot(amp_acc,amp_f, '-o')
plt.ylabel(r'$A_f$')
plt.xlabel(r'$A_{cc}$')
plt.grid()
plt.show()

plt.figure()
plt.plot(amp_acc,amp_omega, '-o')
plt.xlabel(r'$A_{cc}$')
plt.ylabel(r'$\omega_{f}$')

plt.grid()
plt.show()



#
#t = M[0]
#data_ = M[1]
#
#plt.figure()
#plt.plot(t,data_)
#plt.grid()
#plt.show()
#
#
#fu,psd=scipy.signal.welch(data_,fs=df,nperseg=10000)
#
#plt.figure()
#plt.loglog(fu,psd)
#plt.grid()
#plt.show()