# -*- coding: utf-8 -*-
"""
Sea-level Inversion

Calculating ridge production rates from Sea-level Curves, based on papers by: 
    - Gaffin (1987) https://doi.org/10.2475/ajs.287.6.596 
        and 
    - Mills et al (2017) https://doi.org/10.1038/s41467-017-01456-w 
      (simpliefied version of Gaffin's')

Created on Thu Dec 17 11:08:41 2020
@author: datua

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time

start_time = time.time()

def calc_RV(Aocean, dpresent, deltad, Vwater):
    return Aocean * (dpresent + deltad) - Vwater

def calc_Rdest(kdest, RV, RV0):
    return kdest * (RV/RV0)

def dRVdt(RV, t):
    d = np.zeros(RV.size)
    
    for i in range(d.size):
        if i == 0:
            d[i] = (RV[i+1]-RV[i])/(t[i+1]-t[i])
            
        elif i == RV.size-1:
            #d[i] = (RV[i]-RV[i-1])/(t[i]-t[i-1])
            d[i] = d[i-1]
            
        else:
            d[i] = 0.5 * (((RV[i+1]-RV[i])/(t[i+1]-t[i])) + 
                          ((RV[i]-RV[i-1])/(t[i]-t[i-1])))
    
    return d

age = np.load('ridge_production/OrdoSeaLevel.npz')['age']
sealevel = np.load('ridge_production/OrdoSeaLevel.npz')['SeaLevel']

sealevel_km = sealevel/1e3
Aocean = 360e6            # km^2 from Gaffin (1987)
dpresent = 5.4            # km
Vwater = 1.75e9           # km^3
kdest = 3.5               # km^2/yr
RV0 = Aocean*dpresent - Vwater

RV = calc_RV(Aocean, dpresent, sealevel_km, Vwater)
drvdt = dRVdt(RV, -age*1e6)
Rdest = calc_Rdest(kdest, RV, RV0)
Rprod = drvdt + Rdest



plt.subplot(211)
plt.plot(age, sealevel)
plt.subplot(212)
plt.plot(age, Rprod)
plt.show()

print("--- %s seconds---" % (time.time()-start_time))