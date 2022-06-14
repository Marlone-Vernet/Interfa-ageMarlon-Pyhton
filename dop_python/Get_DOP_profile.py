# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:54:47 2020

READ DOP PROFILE 

@author: VERNET
"""
import sys
sys.path.insert(1,'/Users/Gissinger/Desktop/faraday/')


import DOPpy as dop
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

import matplotlib.animation as animation

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib','qt')

plt.rcParams.update({'font.size': 22})

#%%

def find_nearest(array, value,eps): #find nearest element to a given value in an array 
    array = np.asarray(array)
    if array.size == 0:
        return np.nan,np.nan 
    
    idx = (np.abs(array - value)).argmin()
    if np.abs(array[idx] - value)>=eps:
        return np.nan,np.nan
    else:
        return idx,array[idx]

def argmin_(M,al,dx,N_t,N_g,eps,fit_step):
    x_ = dx*np.arange(0,N_g,1)
    x_pos0 =[] #list of the max value
    x_pos1 = [] #list of the value of the corresponding x before the max (at 'mid' height)
    x_pos2 = []
    for k in range(N_t):
        idx = np.where(M[k,:] == max(M[k,:])) #find the indice of the maximum
        h = al*max(M[k,:]) #set the ' mid-height '
        P = np.polyfit(x_,M[k,:],deg=N_g) #get the polynome coefficient degree=10 
        poly = np.poly1d(P) #build a polynomila function with the coefficient 
        x_2 = dx*np.arange(10,idx[0][0],fit_step)
        x_3 = dx*np.arange(idx[0][0],N_g-10,fit_step)
        donne1 = poly(x_2) #data corresponding to the fit before the max
        donne2 = poly(x_3) #data corresponding to the fit after the max
        
        id_1,val1=find_nearest(donne1,h,eps) #find the nearest element
        id_2,va21=find_nearest(donne2,h,eps) #find the nearest element

        x_pos1.append(id_1)
        x_pos2.append(id_2)
        x_pos0.append(idx[0][0])
        if k==2000:
            l1=0.01*0.228*np.arange(1e3,len(donne1)+1e3,1)
            l2=0.01*0.228*np.arange(len(donne1)+1e3,1e3+len(donne1)+len(donne2),1)

            plt.figure()
            plt.plot(l1,donne1, '-o')
            plt.plot(l2,donne2, '-<')
            plt.plot(x_,M[k,:],'--d')    
    
    return x_pos0,x_pos1,x_pos2

def find_nearest_max(M,N_t,width,resolution): #find the max echo near the precedent one (width in mm)
    number_gate=int(width/resolution) #set the number of gate (width of the interval to find the max)
    list_max=[]
    k_=0
    idx0 = np.where(M[k_,:] == max(M[k_,:])) #find the indice of the maximum
    list_max.append(idx0[0][0]) #add the first max in the list
    m_=[max(M[k_,:])]
    k_=1
    while k_<=N_t-1:
        delay=int(number_gate/2) #number of point before and after the former max
        interval=np.arange(list_max[k_-1]-delay,list_max[k_-1]+delay,1) #interval where to look for the max
        idx = np.where(M[k_,interval] == max(M[k_,interval])) #find the indice of the maximum in the interval
        m_.append(max(M[k_,interval]))
        list_max.append(idx[0][0])
        k_+=1

    return np.array(list_max),np.array(m_)


#%%

folder='/Users/Gissinger/Desktop/faraday/'

#data_ = dop.DOP('Ga_Hg_01A.BDD')
data_ = dop.DOP(folder+'test1.BDD')

channel=1
TBP=3.2*1e-3
#TBP=26.8*1e-3

dx=0.228
#
data = data_.getEcho(channel)
N_time,N_gate=np.shape(data)

#data = data_.getVelocity(channel)
#N_time,N_gate=np.shape(data)


time_ = TBP*np.arange(0,len(data),1)


#%%  
""" ECHO """
"""
offset=25 #first gate position in mm
epsilon=1
step=0.01

list0,list1,list2 = argmin_(data,4/5,dx,N_time,N_gate,epsilon,step)

list0 = offset + dx*step*np.array(list0)
list1 = offset + dx*step*np.array(list1)
list2 = offset + dx*step*np.array(list2)

plt.figure()
plt.plot(time_,list0,'-')
plt.grid()
plt.show()

plt.figure()
plt.plot(time_,list1, '-')
plt.plot(time_,list2, '-')
plt.grid()
plt.show()
"""

dim=5 #width of the interval to look for the max in mm
max_,max_v = find_nearest_max(data,N_time,dim,dx)

plt.figure()
plt.plot(time_,dx*max_)
plt.grid()
plt.show()

# plt.figure()
# plt.plot(time_,max_v)
# plt.grid()
# plt.show()

ma=[]
for k in range(N_time):
    idd=np.where(data[k,:] == max(data[k,:]))
    ma.append(idd[0][0])
plt.figure()
plt.plot(time_,dx*np.array(ma))
plt.grid()
plt.show()



f,psd=scipy.signal.welch(ma,fs=1/TBP,nperseg=250)

plt.figure()
plt.loglog(f,psd)
plt.grid()
plt.show()

#%%
""" Anim """

x_ = dx*np.arange(0,N_gate,1)


fig,ax = plt.subplots() # initialise la figure
line1, = plt.plot(x_,data[0,:], '-o') 
line2, = plt.plot([dx*max_[0],dx*max_[0]],[0,2000],'--')
plt.grid()
plt.xlabel('depth [mm]')
plt.xlim(min(x_),max(x_))
plt.ylim(0,2000)



# fonction à définir quand blit=True
# crée l'arrière de l'animation qui sera présent sur chaque image

def animate(i): 
    line1.set_data(x_, data[i,:])
    line2.set_data([dx*max_[i],dx*max_[i]],[0,2000])
    return line1,line2,
 
ani = animation.FuncAnimation(fig, animate, frames=N_time, blit=True, interval=80, repeat=False)

plt.show()

#%%
"""
x_ = dx*np.arange(0,N_gate,1)


fig,ax = plt.subplots() # initialise la figure
line1, = ax[0].plot(x_,echo[0,:], '-o') 
ax[0].grid()
ax[0].xlim(0,30)
ax[0].ylim(0,1000)

line2, = ax[1].plot(echo[0,0])
ax[1].xlim(0,max(time_))

# fonction à définir quand blit=True
# crée l'arrière de l'animation qui sera présent sur chaque image

def animate(i): 
    line1.set_data(x_, echo[i,:])
    line2.set_data(time_[:i],list0[:i])
    return line1,line2,
 
ani = animation.FuncAnimation(fig, animate, frames=N_time, blit=True, interval=20, repeat=False)

plt.show()
"""