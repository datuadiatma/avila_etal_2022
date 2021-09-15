# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 15:00:51 2021

@author: adiatma.1
"""
import numpy as np
import json
from tqdm import tqdm

from scipy.interpolate import interp1d

# Steady state isotopic mass balance equation
def sim_stead_sr(jr, rr, jh, rh):
    rsw = (jr*rr + jh*rh) / (jr+jh)
    return rsw

def run_sim_steady_state_vec(parameter, target_array, tolerance=2e-4,
                         mode='random',
                         riverine_flux=[],
                         riverine_age=[],
                         hydrothermal_flux=[],
                         hydrothermal_age=[],
                         verbose=False):
    """
    function to run a stochastic model with a single dictionary as input

    Paramater
    ---------
    jsonfile : string
        url of json file containing parameters
    """

    if type(parameter) == dict:
        param = parameter
    else:
        with open(parameter) as f:
            param = json.load(f)

    if type(target_array) == dict:
        target = target_array
    else:
        with open(target_array) as k:
            target = json.load(k)

    tmin = param['tmin']
    tmax = param['tmax']
    nt = param['nt']
    age = np.linspace(tmin, tmax, nt)

    s = param['sampling']

    # Resample target array
    resample = interp1d(target['age'], target['sr'])
    target_sr = resample(age)
    # reshape target Sr
    target_sr = (np.zeros((s,nt)) + target_sr).T

    if mode=='random':
        # Generate range of forcing values to sample
        Jriv_range = (np.zeros((nt,s)) +
                      np.random.uniform(param['Jriv'][0], param['Jriv'][1],s))
        
        Rriv_range = (np.zeros((nt,s)) +
                      np.random.uniform(param['Rriv'][0], param['Rriv'][1],s))
        
        Jh_range = (np.zeros((nt,s)) +
                    np.random.uniform(param['Jh'][0], param['Jh'][1], s))
        
        Rh_range = (np.zeros((nt,s)) +
                    np.random.uniform(param['Rh'][0], param['Rh'][1], s))

        # Empty list to store results
        Jriv_res = np.zeros((nt,s))
        Rriv_res = np.zeros((nt,s))
        Jh_res = np.zeros((nt,s))
        Rh_res = np.zeros((nt,s))

        Rsw = sim_stead_sr(Jriv_range, Rriv_range, Jh_range, Rh_range)

        # Store values
        Jriv_res = np.where(np.abs(Rsw - target_sr)<tolerance, Jriv_range, 0)
        Rriv_res = np.where(np.abs(Rsw - target_sr)<tolerance, Rriv_range, 0)
        Jh_res = np.where(np.abs(Rsw - target_sr)<tolerance, Jh_range, 0)
        Rh_res = np.where(np.abs(Rsw - target_sr)<tolerance, Rh_range, 0)

        # Store results as dict
        results = {
                    'Jriv':Jriv_res,
                    'Rriv':Rriv_res,
                    'Jh':Jh_res,
                    'Rh':Rh_res,
                    'age':age
                    }

    elif mode=='riverine':

        f = interp1d(riverine_age, riverine_flux)
        Jriv_input = f(age)
        Jriv_input = (np.zeros((s,nt)) + Jriv_input).T

        # Generate range of forcing values to sample
        Rriv_range = (np.zeros((nt,s)) +
                      np.random.uniform(param['Rriv'][0], param['Rriv'][1],s))
        
        Jh_range = (np.zeros((nt,s)) +
                    np.random.uniform(param['Jh'][0], param['Jh'][1], s))
        
        Rh_range = (np.zeros((nt,s)) +
                    np.random.uniform(param['Rh'][0], param['Rh'][1], s))

        # Empty list to store results
        Jriv_res = np.zeros((nt,s))
        Rriv_res = np.zeros((nt,s))
        Jh_res = np.zeros((nt,s))
        Rh_res = np.zeros((nt,s))
        
        
        Rsw = sim_stead_sr(Jriv_input, Rriv_range, Jh_range, Rh_range)

        # Store values
        Jriv_res = np.where(np.abs(Rsw - target_sr)<tolerance, Jriv_input, 0)
        print('Filtering riverine flux is done')
        Rriv_res = np.where(np.abs(Rsw - target_sr)<tolerance, Rriv_range, 0)
        print('Filtering riverine ratio is done')
        Jh_res = np.where(np.abs(Rsw - target_sr)<tolerance, Jh_range, 0)
        print('Filtering hydrothermal flux is done')
        Rh_res = np.where(np.abs(Rsw - target_sr)<tolerance, Rh_range, 0)
        print('Filtering hydrothermal ratio is done')

        # Store results as dict
        print('MC simulation with riverine input is done')
        results = {
                    'Jriv':Jriv_res,
                    'Rriv':Rriv_res,
                    'Jh':Jh_res,
                    'Rh':Rh_res,
                    'age':age
                    }
    
    elif mode=='hydrothermal':

        f = interp1d(hydrothermal_age, hydrothermal_flux)
        Jh_input = f(age)

        # Generate range of forcing values to sample
        Jriv_range = np.random.uniform(param['Jriv'][0], param['Jriv'][1],s)
        Rriv_range = np.random.uniform(param['Rriv'][0], param['Rriv'][1],s)
        Rh_range = np.random.uniform(param['Rh'][0], param['Rh'][1], s)


        # Empty list to store results
        Jriv_res = np.zeros((nt,s))
        Rriv_res = np.zeros((nt,s))
        Jh_res = np.zeros((nt,s))
        Rh_res = np.zeros((nt,s))
    
        for j in range(nt):
            if verbose:
                print('run sim at %.2f Ma'%age[j])
        # Loop over possible forcing values
            for i in range(s):
                # Riverine flux
                Jriv = Jriv_range[i]
        
                # Hydrothermal flux
                Jh = Jh_input[j]
                # Perturb hydrothermal flux
        
                # Riverine isotopic ratio
                Rriv = Rriv_range[i]
        
                # Hydrothermal isotopic ratio
                Rh = Rh_range[i]
                
                # Simulate
                Rsw = sim_stead_sr(Jriv, Rriv, Jh, Rh)
        
                # Store results if match target
                if np.mean(np.abs(Rsw-target_sr[j])) < tolerance:
                    
                    Jriv_res[j,i] = Jriv
                    Rriv_res[j,i] = Rriv
                    Jh_res[j,i] = Jh
                    Rh_res[j,i] = Rh
    
    
        # Store results as dict
        print('MC simulation with hydrothermal input is done')
        results = {
                    'Jriv':Jriv_res,
                    'Rriv':Rriv_res,
                    'Jh':Jh_res,
                    'Rh':Rh_res,
                    'age':age
                    }

    elif mode=='both':

        f = interp1d(riverine_age, riverine_flux)
        Jriv_input = f(age)

        f = interp1d(hydrothermal_age, hydrothermal_flux)
        Jh_input = f(age)

        # Generate range of forcing values to sample
        Rriv_range = np.random.uniform(param['Rriv'][0], param['Rriv'][1],s)
        Rh_range = np.random.uniform(param['Rh'][0], param['Rh'][1], s)


        # Empty list to store results
        Jriv_res = np.zeros((nt,s))
        Rriv_res = np.zeros((nt,s))
        Jh_res = np.zeros((nt,s))
        Rh_res = np.zeros((nt,s))
    
        for j in range(nt):
            if verbose:
                print('run sim at %.2f Ma'%age[j])
        # Loop over possible forcing values
            for i in range(s):
                # Riverine flux
                Jriv = Jriv_input[j]
        
                # Hydrothermal flux
                Jh = Jh_input[j]
                # Perturb hydrothermal flux
        
                # Riverine isotopic ratio
                Rriv = Rriv_range[i]
        
                # Hydrothermal isotopic ratio
                Rh = Rh_range[i]
                
                # Simulate
                Rsw = sim_stead_sr(Jriv, Rriv, Jh, Rh)
        
                # Store results if match target
                if np.abs(Rsw-target_sr[j]) < tolerance:
                    
                    Jriv_res[j,i] = Jriv
                    Rriv_res[j,i] = Rriv
                    Jh_res[j,i] = Jh
                    Rh_res[j,i] = Rh
    
    
            # Store results as dict
        results = {
                    'Jriv':Jriv_res,
                    'Rriv':Rriv_res,
                    'Jh':Jh_res,
                    'Rh':Rh_res,
                    'age':age
                    }

    return results