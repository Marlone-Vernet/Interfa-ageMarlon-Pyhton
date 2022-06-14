# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import numpy as np 
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
plt.rcParams['font.size']=22

#%%

def g(x):
    return np.sin(x)

def find_local_interval(M):
    inter = []
    bool_true = []
    m = max(M)
    inf = 0
    sup = 0
    while sup<len(M):
        if abs(M[sup])>0.8*m:
            inter.append(sup)
            sup+=1
        else:
            inf = sup
            sup+=1
            if not inter:
                continue
            else:
                bool_true.append(inter)
            inter = []
    return bool_true 

def interval_max(M):
    m_ = max(abs(M))
    idx = np.where(M == m_)
    return idx[0]

def idx_extr(list_of_interval):
    lin = len(list_of_interval)
    list_returned = []
    for k in range(lin):
        list_returned.append( interval_max( list_of_interval[k] ) )
    return np.array(list_returned)



#%% 
x = np.arange(0,10,0.1)
test = find_local_interval(g(x))
test2 = find_local_extremum(g(x), 0.8)

print(test)
LL = g(x)

a = idx_extr(LL[test[0][:]])

plt.figure()
plt.plot(x,g(x))
# [plt.plot(x[test[i][0]], g(x[test[i][0]]), 'o') for i in range(len(test))]
[plt.plot(x[[16, 43, 79]], g(x[[16, 43, 79]]), 'o') for i in range(len(test))]
plt.show()

