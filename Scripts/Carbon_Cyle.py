# -*- coding: utf-8 -*-
"""
Carbon Cycle Model

@author: Datu Adiatma
"""
# Import python packages
from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
lowess = sm.nonparametric.lowess

def Mop(Fwp, Fbp):
    '''
    Mop : Mass of phosphate in the oceans
    Fwp : Flux of phosphate into the oceans
    Pbp : flux of phosphate out of the oceans due to burial
    dt  : time interval
    y   : dummy variable to store results
    '''
    y = (Fwp - Fbp)
    return y

def Fwp(weatherability, ClimateWeatheringFactor):
    '''
    Flux of phosphate to the oceans (10^12 mol per ky)
    '''
    y = 30 * weatherability * ClimateWeatheringFactor
    return y

def Fbp(Mop, Mop_initial):
    y = 30 * (Mop / Mop_initial)
    return y

def Moc(Fworg, Fvolc, Fborg, Fwsil):
    y = (Fworg + Fvolc - Fborg - Fwsil)
    return y

def Fworg(weatherability, ClimateWeatheringFactor):
    y = 1e4 * weatherability * ClimateWeatheringFactor
    return y

def Fborg(Fbp, CtoP=333):
    y = Fbp * CtoP
    return y

def Fwsil(weatherability, ClimateWeatheringFactor):
    y = 6e3 * weatherability * ClimateWeatheringFactor
    return y

def Ro(ro, rb, rr, Fwsr, Fvolc, fhyd, Mosr):
    y = (((1+ro)/(1+rb)) * (rr-ro) * Fwsr + ((1+ro)/(1+rb)) * (rb-ro) * Fvolc * fhyd) / (Mosr)
    return y

def Rr(time, weatherability):
    if time < 2e4:
        y = 0.7106 - time * 1e-8
    else:
        y = (0.7104 + (weatherability - 1) * 0.7043) / (weatherability)
    return y

def Fwsr(weatherability, climateweatheringfactor):
    y = 30 * weatherability * climateweatheringfactor
    return y

def pCO2(Moc, Moc_0=1.61e7, pCO2_0=5000):
    y = (Moc / Moc_0)**2 * pCO2_0
    return y

def CWF(pCO2, pCO2_0=5000):
    y = (pCO2/pCO2_0)**0.3
    return y

dt = 1

# time
time_min = 0
time_max = 40000
time = np.arange(time_min, time_max, dt)

# array size
t = len(time)

age = 484 - time/1e3

#set time and rate of weatherability change
rise_start = np.where(np.round(age)==462)[0][0]     #statement looks up timestep closest to specified age, e.g. 463 Ma
rise_stop = np.where(np.round(age)==455)[0][0]      #original Young model has rise start at 463, stop 459, fall 447, stop 443
fall_start = np.where(np.round(age)==453)[0][0]

# Weatherability
W = np.ones(t)
W[rise_start:rise_stop] = np.linspace(1.0, 1.25, (rise_stop-rise_start))
W[rise_stop:] = 1.25

# pCO2
pco2= np.ones(t) # array for pCO2
pco2_0 = 5000     # initial value of pCO2
pco2[0] = pco2_0

# Climate Weathering Factor (CWP)
cwf = np.ones(t)
cwf[0] = CWF(pco2[0])

# load sea level data
ordosealevel = np.load("../Data/OrdoSeaLevel.npz")['SeaLevel']
ordoseaage = np.load("../Data/OrdoSeaLevel.npz")['age']

# resample data to fit our modeling array
from scipy import interpolate
f = interpolate.interp1d(ordoseaage, ordosealevel)
sealevel = f(age)

# normalize sea level relative to mean
sl_norm = sealevel / sealevel.mean()

# Initialize Fvolc; also tuning of volc flux
fvolc = np.ones(t)
fvolc = sl_norm * 6000 # initial value = 6000 (Kump and Arthur, 1999)


#Initialize arrays for diff.eq #1
mop = np.ones(t)  # array for Mop
mop_0 = 3e3       # initial value of Mop
mop[0] = mop_0    # assign initial value into array

fwp = Fwp(W, cwf)     # initiate array for phosphorus flux
fbp = Fbp(mop, mop_0) # initiate array for phosphorus burial

# Initialize arrays for diff.eq #2
moc = np.ones(t)  # array for Moc
moc_0 = 1.61e7    # initial value of Moc
moc[0] = moc_0    # assign initial value into array

fworg = Fworg(W, cwf)
fborg = Fborg(fbp)
fwsil = Fwsil(W, cwf)

# Model implementation
for i in range(t-1):
    
    fworg[i+1] = Fworg(W[i], cwf[i])
    fborg[i+1] = Fborg(fbp[i])
    fwsil[i+1] = Fwsil(W[i], cwf[i])
    moc[i+1] = moc[i] + dt*Moc(fworg[i], fvolc[i], fborg[i], fwsil[i])
    
    pco2[i+1] = pCO2(moc[i], moc_0, pco2_0)
    cwf[i+1] = CWF(pco2[i])
    
    fwp[i+1] = Fwp(W[i], cwf[i])
    fbp[i+1] = Fbp(mop[i], mop_0)
    mop[i+1] = mop[i] + dt*Mop(fwp[i],fbp[i])

# Store output into csv file
output = {'time':time,
          'age':age,
          'W' : W,
          'fvolc': fvolc,
          'fworg':fworg,
          'fborg':fborg,
          'fwsil':fwsil,
          'moc':moc,
          'pco2':pco2,
          'cwf':cwf,
          'fwp':fwp,
          'fbp':fbp,
          'mop':mop}

#output_df = pd.DataFrame(data=output)
#output_df.to_csv('Output/young_etal_ouput.csv', index=None)

# Plotting
fig, (ax1, ax3) = plt.subplots(2, 1, sharex=True, figsize=(8,10))

ax1.plot(age, W, 'k-', label='Weatherability')
ax1.legend(bbox_to_anchor=(0.3, 0.35), edgecolor='None')
ax1.set_ylabel('Weatherability', fontsize=14)

ax3.plot(age, pco2, 'r--', label= '$pCO_2$ Model')
ax3.set_ylabel('$pCO_2$(ppmv)', fontsize=14)
ax3.legend(bbox_to_anchor=(0.3, 0.35), edgecolor='None')

ax4 = ax3.twinx()
ax4.plot(age, fvolc, 'g-', label='Volcanism')
ax4.plot(age, fwsil, 'y:', label='Silicate Weathering Flux')
ax4.set_ylabel('Volcanism and Silicate Weatheing\n$10^{12}$ mol carbon/ky', fontsize=14)
ax4.legend(bbox_to_anchor=(0.43, 0.25), edgecolor='None')
ax4.set_ylim(4000, 7600)


ax3.set_xlabel('Age (Myr)', fontsize=14)
ax3.set_xlim(max(age), min(age))

plt.tight_layout()
plt.show()
