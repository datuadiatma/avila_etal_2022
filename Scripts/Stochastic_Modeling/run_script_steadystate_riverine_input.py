# -*- coding: utf-8 -*-
"""
Created on Thu May 27 12:58:39 2021
Python script to run "monte-carlo-optimized" strontium box model
@author: adiatma.1
"""
from mcsr import run_sim
from mcsr_vec import run_sim_steady_state_vec as rss_vec

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

# Load riverine flux at 40% increase in weatherability
riverine = pd.read_csv('riverine.csv')

# Load hydrothermal
ht_flux = np.load("OrdoSeaLevel_Hydrothermal.npz")['Jh']
ht_age = np.load("OrdoSeaLevel_Hydrothermal.npz")['age']



starttime = time.time()

np.random.seed(614)

par = rss_vec('param_steady.json', 'target.json', tolerance=2.5e-5,
          mode='riverine',
          riverine_flux=riverine['friv'],
          riverine_age=riverine['age'],
          hydrothermal_flux=ht_flux,
          hydrothermal_age=ht_age)

exectime = time.time() - starttime
print('Execution time: %.1f s'%exectime)

Jriv = par['Jriv']
Rriv = par['Rriv']
Jh = par['Jh']
Rh = par['Rh']
age = par['age']

non_zero = np.where(Jriv!=0)

a, b = Jriv.shape

# Age array
age_h = np.zeros((b, a)) + age
age_h = age_h.T

age_flat = age_h[non_zero].flatten()
Rh_flat = Rh[non_zero].flatten()
Rriv_flat = Rriv[non_zero].flatten()

Rsw = np.zeros_like(Jriv)
Rsw[non_zero] = ((Jriv[non_zero]*Rriv[non_zero] + Jh[non_zero]*Rh[non_zero])
                 / (Jriv[non_zero] + Jh[non_zero]))

Jriv_mean = np.zeros_like(age)
Jriv_stdev = np.zeros_like(age)
Jh_mean = np.zeros_like(age)
Jh_stdev = np.zeros_like(age)
Rriv_mean = np.zeros_like(age)
Rriv_stdev = np.zeros_like(age)
Rh_mean = np.zeros_like(age)
Rh_stdev = np.zeros_like(age)

Rsw_mean = np.zeros_like(age)
Rsw_stdev = np.zeros_like(age)

for i in range(len(age)):
    Jriv_d = Jriv[i,:]
    Jriv_mean[i] = np.mean(Jriv_d[Jriv_d!=0])
    Jriv_stdev[i] = np.std(Jriv_d[Jriv_d!=0])

    Rriv_d = Rriv[i,:]
    Rriv_mean[i] = np.mean(Rriv_d[Rriv_d!=0])
    Rriv_stdev[i] = np.std(Rriv_d[Rriv_d!=0])

    Jh_d = Jh[i,:]
    Jh_mean[i] = np.mean(Jh_d[Jh_d!=0])
    Jh_stdev[i] = np.std(Jh_d[Jh_d!=0])

    Rh_d = Rh[i,:]
    Rh_mean[i] = np.mean(Rh_d[Rh_d!=0])
    Rh_stdev[i] = np.std(Rh_d[Rh_d!=0])

    Rsw_d = Rsw[i,:]
    Rsw_mean[i] = np.mean(Rsw_d[Rsw_d!=0])
    Rsw_stdev[i] = np.std(Rsw_d[Rsw_d!=0])


# Error band
Jriv_hi = Jriv_mean + Jriv_stdev
Jriv_lo = Jriv_mean - Jriv_stdev
Rriv_hi = Rriv_mean + Rriv_stdev
Rriv_lo = Rriv_mean - Rriv_stdev

Jh_hi = Jh_mean + Jh_stdev
Jh_lo = Jh_mean - Jh_stdev
Rh_hi = Rh_mean + Rh_stdev
Rh_lo = Rh_mean - Rh_stdev

Rsw_hi = (Jriv_hi*Rriv_hi + Jh_hi*Rh_hi) / (Jriv_hi + Jh_hi)
Rsw_lo = (Jriv_lo*Rriv_lo + Jh_lo*Rh_lo) / (Jriv_lo + Jh_lo)

# Store output as csv
results = {
    'age' : age,
    'Jriv' : Jriv_mean,
    'Jriv_hi' : Jriv_hi,
    'Jriv_lo' : Jriv_lo,
    'Rriv' : Rriv_mean,
    'Rriv_hi' : Rriv_hi,
    'Rriv_lo' : Rriv_lo,
    'Jh' : Jh_mean,
    'Jh_hi' : Jh_hi,
    'Jh_lo' : Jh_lo,
    'Rh' : Rh_mean,
    'Rh_hi' : Rh_hi,
    'Rh_lo' : Rh_lo
    }

# Run transient model
nt = len(age)
dt = (age.max() - age.min())*1e6/nt
n = np.ones(nt)*1.9e17

Rsw_transient = np.zeros(nt)
Rsw_transient_hi = np.zeros(nt)
Rsw_transient_lo = np.zeros(nt)

Rsw_transient = run_sim(nt, dt, age, Jriv_mean, Rriv_mean, Rsw_transient,
                        Jh_mean, Rh_mean, n)
Rsw_transient_hi = run_sim(nt, dt, age, Jriv_hi, Rriv_hi, Rsw_transient_hi, Jh_hi,
                        Rh_hi, n)

Rsw_transient_lo = run_sim(nt, dt, age, Jriv_lo, Rriv_lo, Rsw_transient_lo, Jh_lo,
                        Rh_lo, n)

df = pd.read_excel('mastersr.xlsx')
dx = pd.read_csv('../../Output/SL_Model_output.csv')

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8,10))

ax1.plot(age, Jriv_mean, 'b--', alpha=0.5, label='Riverine\n(Weatherability=1.4)')
ax1.fill_between(age, Jriv_hi, Jriv_lo, fc='blue', alpha=0.15)

ax1.plot(age, Jh_mean, 'r--', alpha=0.5, label='Hydrothermal')
ax1.plot(ht_age, ht_flux, 'r-', alpha=0.5, label='Sealevel-inverted\nHydrothermal')
ax1.fill_between(age, Jh_hi, Jh_lo, fc='red', alpha=0.15)
ax1.set_ylabel('Sr Flux', fontsize=14)
ax1.legend(loc="upper left")

ax2.plot(age, Rriv_mean, 'b--', alpha=0.5, label='Riverine')
ax2.fill_between(age, Rriv_hi, Rriv_lo, fc='blue', alpha=0.15)
ax2.set_ylabel(r'$^{87}Sr/^{86}Sr$', fontsize=14)

ax2.plot(age, Rh_mean, 'r--', alpha=0.5, label='Hydrothermal')
ax2.fill_between(age, Rh_hi, Rh_lo, fc='red', alpha=0.15)
ax2.legend()

ax3.plot(age, Rsw_mean, c='green', ls='-', label='Model')
ax3.fill_between(age, Rsw_hi, Rsw_lo, fc='green', alpha=0.15)
ax3.scatter(df['age'], df['sr'], fc='green', ec='black', label='Conodont Sr',
            alpha=0.5)
ax3.set_xlim(480, 455)
ax3.set_ylabel(r'$^{87}Sr/^{86}Sr_{seawater}$')
ax3.set_xlabel('Age (Ma)')
ax3.legend(loc = 'lower left')

plt.tight_layout()
plt.savefig('../../Figures/MC_with_riverine_input.png', dpi=300)
plt.savefig('../../Figures/MC_with_riverine_input.svg')


# Grid Spec plotting
fig2 = plt.figure(constrained_layout=True, figsize = (10, 8))
gs = fig2.add_gridspec(2, 2)
ag1 = fig2.add_subplot(gs[0,0:])
ag1.plot(age, Rsw_transient, c='black', ls='--', label='Updated Model')
ag1.plot(dx['age'], dx['Rsw'], c='steelblue', ls='--',
        label='Hydrothermal-driven\nModel')
ag1.fill_between(age, Rsw_hi, Rsw_lo, fc='green', alpha=0.15)
ag1.scatter(df['age'], df['sr'], fc='green', ec='black', label='Conodont Sr',
            alpha=0.5)
ag1.set_xlim(480, 450)
ag1.set_ylabel(r'$^{87}Sr/^{86}Sr_{seawater}$', fontsize=14)
ag1.set_xlabel('Age (Ma)')
ag1.legend(loc = 'lower left')


ag2 = fig2.add_subplot(gs[1,0])
ag2.plot(age, Jriv_mean, 'b--', alpha=0.5, label='Riverine\n(Weatherability=1.4)')
ag2.fill_between(age, Jriv_hi, Jriv_lo, fc='blue', alpha=0.15)
ag2.plot(age, Jh_mean, 'r--', alpha=0.5, label='Hydrothermal')
ag2.plot(ht_age, ht_flux, 'r-', alpha=0.5, label='Sealevel-inverted\nHydrothermal')
ag2.fill_between(age, Jh_hi, Jh_lo, fc='red', alpha=0.15)
ag2.set_ylabel('Sr Flux', fontsize=14)
ag2.legend(loc="upper left")
ag2.set_xlim(480, 450)
ag2.set_xlabel('Age (Ma)')

ag3 = fig2.add_subplot(gs[1,1])
ag3.plot(age, Rriv_mean, 'b--', alpha=0.5, label='Riverine')
ag3.fill_between(age, Rriv_hi, Rriv_lo, fc='blue', alpha=0.15)
ag3.set_ylabel(r'$^{87}Sr/^{86}Sr_{fluxes}$', fontsize=14)
ag3.plot(age, Rh_mean, 'r--', alpha=0.5, label='Hydrothermal')
ag3.fill_between(age, Rh_hi, Rh_lo, fc='red', alpha=0.15)
ag3.legend()
ag3.set_xlim(480, 450)
ag3.set_xlabel('Age (Ma)')

plt.savefig('../../Figures/MC_Transient_Model.png', dpi=300)
plt.savefig('../../Figures/MC_Transient_Model.svg')
plt.show()