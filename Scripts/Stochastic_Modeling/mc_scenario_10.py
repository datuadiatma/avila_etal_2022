# -*- coding: utf-8 -*-
"""
Created on Tue Septh 14 2021
Python script to run "monte-carlo-optimized" strontium box model
Scenario 10: modern hydrothermal flux


@author: adiatma.1
"""
# Import modules and libraries
# ----------------------------

# Import time to keep track of timing
from time import time

# Import the trifecta of python scientific computation module
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import module to run monte carlo simulation
from mcsr import run_sim
from mcsr_vec import run_sim_steady_state_vec as rss_vec

# Monte Carlo Parameters
# ----------------------

# Set random seed for reproducibility
np.random.seed(614)

# Adjust Jh paramater range of published modern values

mc_parameter = {
    "tmin"     :  480,
    "tmax"     :  450,
    "nt"       :  700,

    "Jriv"     :  [10e9, 190e9],
    "Rriv"     :  [0.7025, 0.7120],

    "Jh"       :  [4.5e9, 4e10],
    "Rh"       :  [0.7030, 0.7070],

    "sampling" : 80000
}
# Start time
starttime = time()

# run MC resampling
par = rss_vec(mc_parameter, 'target.json', 1e-5,
              'random', verbose=True)

# Unpack results into variables
Jriv = par['Jriv']
Rriv = par['Rriv']
Jh = par['Jh']
Rh = par['Rh']
age = par['age']

# Array selector to choose non_zero values
non_zero = np.where(Jriv!=0)

# Calculate Rsw steady state solution
Rsw = np.zeros_like(Jriv)
Rsw[non_zero] = ((Jriv[non_zero]*Rriv[non_zero] + Jh[non_zero]*Rh[non_zero])
                 / (Jriv[non_zero] + Jh[non_zero]))

# Calculate mean and standard deviation of solution space
# -------------------------------------------------------

# Create empty array to store solutions
Jriv_mean = np.zeros_like(age)
Jriv_stdev = np.zeros_like(age)
Jriv_max = np.zeros_like(age)
Jriv_min = np.zeros_like(age)

Jh_mean = np.zeros_like(age)
Jh_stdev = np.zeros_like(age)
Jh_max = np.zeros_like(age)
Jh_min = np.zeros_like(age)

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

# Run transient model
nt = len(age)
dt = (age.max() - age.min())*1e6/nt
n = np.ones(nt)*1.9e17

Rsw_transient = np.zeros(nt)
Rsw_transient_hi = np.zeros(nt)
Rsw_transient_lo = np.zeros(nt)

Rsw_transient = run_sim(nt, dt, age, Jriv_mean, Rriv_mean, Rsw_transient,
                        Jh_mean, Rh_mean, n)

Rsw_transient_hi = run_sim(nt, dt, age, Jriv_hi, Rriv_hi, Rsw_transient_hi,
                           Jh_hi, Rh_hi, n)

Rsw_transient_lo = run_sim(nt, dt, age, Jriv_lo, Rriv_lo, Rsw_transient_lo,
                           Jh_lo, Rh_lo, n)

df = pd.read_excel('mastersr.xlsx')
dx = pd.read_csv('../../Output/SL_Model_output.csv')

# Grid Spec plotting
fig2 = plt.figure(constrained_layout=True, figsize = (10, 8))
gs = fig2.add_gridspec(2, 2)
ag1 = fig2.add_subplot(gs[0,0:])
ag1.plot(age, Rsw_transient, c='k', ls='--', lw=3,
         label='Monte Carlo-optimized\nTransient Box Model')
ag1.plot(dx['age'], dx['Rsw'], c='steelblue', ls='--',
        label='Hydrothermal-driven\nBox Model')
ag1.fill_between(age, Rsw_hi, Rsw_lo, fc='green', alpha=0.15)
ag1.scatter(df['age'], df['sr'], fc='green', ec='black', label='Conodont Sr',
            alpha=0.5)
ag1.set_xlim(480, 450)
ag1.set_ylim(0.7075, 0.7095)
ag1.set_yticks(np.linspace(0.7078, 0.7094, 5))
ag1.set_ylabel(r'$^{87}Sr/^{86}Sr_{seawater}$', fontsize=14)
ag1.set_xlabel('Age (Ma)')
ag1.legend(loc = 'lower left')

ag1s = ag1.twinx()
ag1s.plot(age, n, c='orange',label='Sr Reservoir')
ag1s.set_ylabel('Sr Reservoir\n(mol)', fontsize=14)
ag1s.legend(loc = 'lower right')
ag1s.set_xlim(480, 450)

ag2 = fig2.add_subplot(gs[1,0])
ag2.plot(age, Jriv_mean, 'b--', alpha=0.5, label='Riverine')
ag2.fill_between(age, Jriv_hi, Jriv_lo, fc='blue', alpha=0.15)
ag2.plot(age, Jh_mean, 'r--', alpha=0.5, label='Hydrothermal')
ag2.fill_between(age, Jh_hi, Jh_lo, fc='red', alpha=0.15)
ag2.set_ylabel('Sr Flux', fontsize=14)
ag2.legend(loc="upper left")
ag2.set_xlim(480, 450)
# ag2.set_ylim(0, 9e10)
# ag2.set_yticks(np.linspace(0, 8e10, 5))
ag2.set_xlabel('Age (Ma)')

ag3 = fig2.add_subplot(gs[1,1])
ag3.plot(age, Rriv_mean, 'b--', alpha=0.5, label='Riverine')
ag3.fill_between(age, Rriv_hi, Rriv_lo, fc='blue', alpha=0.15)
ag3.set_ylabel(r'$^{87}Sr/^{86}Sr_{fluxes}$', fontsize=14)
ag3.plot(age, Rh_mean, 'r--', alpha=0.5, label='Hydrothermal')
ag3.fill_between(age, Rh_hi, Rh_lo, fc='red', alpha=0.15)
ag3.legend()
ag3.set_xlim(480, 450)
ag3.set_ylim(0.7030, 0.7125)
ag3.set_yticks(np.linspace(0.7035, 0.7115, 5))
ag3.set_xlabel('Age (Ma)')

# Print execution time
exectime = time() - starttime
print('Execution time: %.1f s'%exectime)

plt.savefig(
    '../../Figures/MonteCarlo_Simulation/scenario10_modern_hydrothermalFlux.png',
    dpi=300
    )

plt.savefig(
    '../../Figures/MonteCarlo_Simulation/scenario10_modern_hydrothermalFlux.svg'
    )

plt.show()