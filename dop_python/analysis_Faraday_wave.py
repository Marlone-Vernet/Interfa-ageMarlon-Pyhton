# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:50:33 2020

analysis of the Faraday wave data 

from capacity probes 

@author: Vernet
"""


import sys
sys.path.insert(1,'/Users/Gissinger/Desktop/python_NI/')

import PyDAQmx as nidaq
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pickle 

import method_pico as pic

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib','qt')

plt.rcParams.update({'font.size': 22})

#%%

file_ = {'08':"C:/Users/Gissinger/Desktop/faraday/080720/",
         '09-1':"C:/Users/Gissinger/Desktop/faraday/090720/",
         '09-2':"C:/Users/Gissinger/Desktop/faraday/090720/",
         '21-1':"C:/Users/Gissinger/Desktop/faraday/210720/",
         '21-2':"C:/Users/Gissinger/Desktop/faraday/210720/",
         '21-3':"C:/Users/Gissinger/Desktop/faraday/210720/",
         '22-1':"C:/Users/Gissinger/Desktop/faraday/220720/",
         '22-2':"C:/Users/Gissinger/Desktop/faraday/220720/",
         '23-1':"C:/Users/Gissinger/Desktop/faraday/230720/",
         '23-2':"C:/Users/Gissinger/Desktop/faraday/230720/",
         '23-3':"C:/Users/Gissinger/Desktop/faraday/230720/",

         }

current_ = {'08':[2,3,4,7,11,14,17,20,23,27,30,34,37,40,45],
            '09-1':[5,10,13,16,20,23,26,30,33,36,40,44],
            '09-2':[3,7,10,12,17,20,23,24,25,27,28,29,30,33,36,39,42],
            '21-1':[2,5,8,11,13,15,17,20,22,24,26,28,30,31,32,33,34,35,36,37,38,41,45],
            '21-2':[3,6,9,12,15,18,20,22,24,26,28,30,32,34,35,37,38,39,40,41,43,46],
            '21-3':[2,6,9,12,16,19,22,24,26,28,33,35,37,39,41,43,46],
            '22-1':[3,11,14,17,23,26,28,30,32,35,38,40,43,46],
            '22-2':[4,8,10,12,14,16,18,20,22,24,26,28,31,33,36,38,40,43,46],
            '23-1':[4,7,10,13,16,19,22,25,28,31,34,37,40,43],
            '23-2':[4,7,11,13,15,17,19,22,24,26,28,30,32,35,38,41,44,47],
            '23-3':[4,8,11,14,17,20,25,28,30,32,34,35,36,38,42,44,47]
            }

cle = ['08','09-1','09-2','21-1','21-2','21-3','22-1','22-2','23-1','23-2','23-3']
pre_ = {'08':'o', '09-1':'', '09-2':'m_', '21-1':'d_','21-2':'m_','21-3':'d_',
        '22-1':'m_','22-2':'d_','23-1':'m_','23-2':'m_','23-3':'d_'}
hert = {'08':'', '09-1':'', '09-2':'', '21-1':'','21-2':'40Hz_','21-3':'40Hz_',
        '22-1':'36Hz_','22-2':'36Hz_','23-1':'46Hz_','23-2':'30Hz_','23-3':'30Hz_'}
hertz = {'08':'30Hz', '09-1':'30Hz', '09-2':'30Hz', '21-1':'30Hz','21-2':'40Hz','21-3':'40Hz',
         '22-1':'36Hz','22-2':'36Hz','23-1':'46Hz','23-2':'30Hz','23-3':'30Hz'}
donne_ = {}

to_load = [9,10]

for k in to_load:
    
    amp_acc = []
    amp_farad = []
    for i in current_[cle[k]]:
        number = i
        name = pre_[cle[k]]+"accelero_"+hert[cle[k]]+str(number)+"dcA.pkl" #data 080720 oaccelero & ofarafay ...
        name_ = pre_[cle[k]]+"faraday_"+hertz[cle[k]]+"_"+str(number)+"dcA.pkl"
        
      
        with open(file_[cle[k]] + name, 'rb') as f:
            a_ = pickle.load(f)
        with open(file_[cle[k]] + name_, 'rb') as f:
            fa_ = pickle.load(f)
    
        amp_acc.append(a_)
        amp_farad.append(fa_)
    
    donne_[cle[k],'accelero'] = amp_acc
    donne_[cle[k],'faraday'] = amp_farad

# sig_acc = np.std(amp_acc,axis=1)
# sig_farad = np.std(amp_farad,axis=1)

#%%
""" signal faraday probably 1V -> 1µV detection synchrone """

fa = 10000

dt = 1/fa
time_ = dt*np.arange(0,len(amp_acc[0]),1)

result = {}
freq = {'08':15, '09-1':15, '09-2':15, '21-1':15,'21-2':20,'21-3':20,'22-1':18,
        '22-2':18,'23-1':24,'23-2':15,'23-3':15} #freq in Hz
to_do = [9,10]

for j in to_do:
    amp = []
    F,PSD = [],[]
    max_ = []
    for k in range(len(current_[cle[j]])):
        f,psd = scipy.signal.welch(donne_[cle[j], 'faraday'][k], fa, nperseg = 10000)
        F.append(f)
        PSD.append(psd)    
        max_.append(psd[int(freq[cle[j]])])
        
        a_rms = np.sqrt(np.mean(np.power(donne_[cle[j], 'accelero'][k], 2)))
        amp.append(a_rms)
    
    result[cle[j], 'PSD'] = PSD
    result[cle[j], 'F'] = F
    result[cle[j], 'max'] = max_
    result[cle[j], 'a_rms'] = amp

# le = len(amp_farad[0])
# time_ = np.arange(0,le,1)/fa
mark = ['-o','-s','-d','-p','-h','*']

plt.figure()
plt.loglog(result[cle[9], 'a_rms'], result[cle[9], 'max'], mark[0], label=r'36 Hz $\nearrow$')
plt.loglog(result[cle[10], 'a_rms'], result[cle[10], 'max'], mark[1], label=r'36 Hz $\searrow$')
plt.xlabel(r'$a_{RMS} [m.s^{-2}]$')
plt.ylabel(r'$S_p(f_d)$')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# plt.figure()
# plt.plot(I,sig_acc,'-o')
# plt.grid()
# plt.show()

# plt.figure()
# plt.plot(sig_acc, sig_farad,'-o')
# plt.grid()
# plt.show()

# for k in range(len(I)):
#     plt.figure(I[k])
#     plt.plot(time_, amp_farad[k], label=str(I[k])+'dcA')
#     plt.xlabel('t [s]')
#     plt.ylabel(r'$\xi [?]$')
#     plt.grid()
#     plt.legend()
#     plt.show()

plt.figure()
[plt.loglog(F[k],PSD[k]) for k in range(len(I))]
plt.grid()
plt.show()


""" produce a nice sinus ;) """ 
# N=50
# exp = mv_average(data2, N)

# N_ = 100
# exp_2 = pic.cleaner(exp, N_)

# plt.figure()
# plt.plot(t[:len(exp_2)],exp_2)
# plt.grid()
# plt.show()

   