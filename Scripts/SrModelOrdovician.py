# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:38:34 2020

@author: datua
"""
# Load modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Functions
# ---------
# Isotopic mass balace equation
def simSr(jr, rr, rsw, jh, rh, n):
    """

    Strontium isotopic mass balance.

    Parameters
    ----------
    jr : float
        Global riverine flux of Sr
    rr : float
        Strontium isotopic ratio of global riverine flux.
    rsw : float
        Strontium isotopic ratio of seawater.
    jh : float
        Hydrothermal flux of Sr.
    rh : float
        Strontium isotopic ratio of hydrothermal flux (jh).
    n : float
        Strontium reservoir size of the ocean.

    Returns
    -------
    rSr : float
        Strontium isotopic ratio of seawater.

    """
    rSr = (jr*(rr-rsw) + jh*(rh-rsw)) / n
    return rSr

# Function to run model
def run_sim(nt, dt, age, jr, rr, rsw, jh, rh, n):
    rsw0 = (jr[0]*rr[0] + jh[0]*rh[0])/(jr[0]+jh[0])
    rsw[0] = rsw0
    
    for i in range(nt-1):
        rsw[i+1] = rsw[i] + simSr(jr[i], rr[i], rsw[i], jh[i], rh[i], n[i])*dt
    
    grad1 = np.diff(rsw)/np.diff(age)
    grad1 = np.append(grad1, grad1[-1])
    grad2 = np.diff(grad1) / np.diff(age)
    grad2 = np.append(grad2, grad2[-1])
    
    return rsw, grad1

# Define array of time
# --------------------
tmin = 487      # ~ base of Ordovician in Ma (GTS2020)
tmax = 443      # ~ base of Silurian in Ma (GST2020)
nt = 100000
dt = (tmin - tmax)*1e6 / nt
time = np.linspace(0, (tmin-tmax)*1e6, nt)
age = np.linspace(tmin, tmax, nt)

# Initial values and paramaters
# -----------------------------
# Riverine flux
Jriv0 = 2.5e10
Jriv = np.ones(nt) * Jriv0

# Riverine isotopic ratio
Rriv0 = 0.7119
Rriv = np.ones(nt) * Rriv0

# Axial (crestal) hydrothermal flux
Jhcrest0 = 9.00e9
Jhcrest = np.ones(nt) * Jhcrest0

# Off-axis (flank) hydrothermal flux
Jhflank0 = 5.00e9
Jhflank = np.ones(nt) * Jhflank0

# Total hydrothermal flux
Jh0 = Jhflank0 + Jhcrest0
Jh = np.zeros(nt)
Jh = Jhflank + Jhcrest

# Hydrothermal isotopic ratio
Rh0 = 0.7037
Rh = np.ones(nt) * Rh0

# Array to store isotopic ratio of seawater and rate of change
Rsw = np.zeros(nt)
GradSr = np.zeros(nt)

# Reservoir size
N = np.ones(nt) * 1.9e17

# System perturbation
# -------------------
# timing of perturbation in Ma
start_age = 463
stop_age = 455

# convert age to array index
start = np.where(np.round(age)==start_age)[0][0]
stop = np.where(np.round(age)==stop_age)[0][0]

# perturb the system by increasing hydrothermal flux
f = 1.5
Jh[start:stop] = np.linspace(Jh0, Jh0*f, (stop-start))
Jh[stop:] = Jh0 * f

# Run the model
# -------------

Rsw, GradSr = run_sim(nt, dt, age, Jriv, Rriv, Rsw, Jh, Rh, N)

# Load sr measurement for comparison
SrOrdo = pd.read_csv('../Data/Arbuckle_Sr.csv')

# plot the result
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax1.plot(age, Rsw, label="Sr model")
ax1.scatter(SrOrdo['Age20'], SrOrdo['Sr'], label='Sr Conodont', alpha=0.4)
ax1.set_ylabel('$^{87}Sr/^{86}Sr$')
ax1.legend(loc='right')

ax1b = ax1.twinx()
ax1b.plot(age, GradSr, color='green',label='Rate of Change')
ax1b.set_ylabel('Rate of Change\n(/yr)')
ax1b.legend(loc='upper left')

ax2.plot(age, Jh, c='maroon',label='hydrothermal flux')
ax2.set_ylabel('mol/yr')
ax2.set_xlabel('Age (Ma)')
ax2.legend(loc='right')

plt.tight_layout()
plt.show()
plt.savefig('../Figure/OrdovicianSr.png')