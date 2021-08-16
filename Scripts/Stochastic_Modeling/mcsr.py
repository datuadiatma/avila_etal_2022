"""
mcsr is a monte carlo implementation of sr isotopic mass balance box modeling
in python

author : Y. Datu Adiatma
github : github.com/datuadiatma

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import json

from tqdm import tqdm

# Mass balance
def simNSr(jr, jh, jcarb):
    """
    Strontium Mass Balance

    Parameters
    ----------
    jr : float
        Global riverine flux of Sr
    jh : float
        Hydrothermal flux of Sr.
    jcarb : float
        strontium uptake during carbonate deposition
    
    Returns
    -------
    nsr : float
        Seawater Sr reservoir size in mol
    """
    nsr = jr + jh - jcarb
    return nsr

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

# Steady state isotopic mass balance equation
def sim_stead_sr(jr, rr, jh, rh):
    rsw = (jr*rr + jh*rh) / (jr+jh)
    return rsw

# Function to run model
def run_sim(nt, dt, age, jr, rr, rsw, jh, rh, n):
    """  
    Solving diff. equations defined in simSr() using forward Euler method.

    Parameters
    ----------
    nt : int
        number of time steps to run model
    dt : float
        the size of each time step
    age : float
        age in million years
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
    rSw : float
        Strontium isotopic ratio of seawater.
     
    """
    rsw0 = (jr[0]*rr[0] + jh[0]*rh[0])/(jr[0]+jh[0])
    rsw[0] = rsw0

    jcarb0 = jr[0] + jh[0]
    k = jcarb0 / n[0]
    jcarb = jcarb0

    for i in range(nt-1):
        n[i+1] = n[i] + simNSr(jr[i], jh[i], jcarb)*dt
        rsw[i+1] = rsw[i] + simSr(jr[i], rr[i], rsw[i], jh[i], rh[i], n[i])*dt
        jcarb = k * n[i]


    return rsw

def perturb(flux, age, age_min, age_max, factor):
    """
    Function to generate flux perturbation

    Parameters
    ----------
    flux : array
        flux that will be perturbed
    age : array
        array containing age in million years
    age_min : float
        starting point of perturbation
    age_max : float
        stopping point of perturbation, anything beyond this will be constant
    factors: float
        multpying factor to perturbate the flux
    
    Returns
    -------
    flux : array
        array of flux that has been perturbed
    """
    # Array index to start and stop
    start = np.where(np.round(age)==age_min)[0][0]
    stop = np.where(np.round(age)==age_max)[0][0]

    flux[start:stop] = np.linspace(flux[start], flux[start]*factor, stop-start)
    flux[stop:] = flux[start]*factor

    return flux

def perturb_ratio(ratio, age, age_min, age_max, factor):
    """
    Function to generate flux perturbation

    Parameters
    ----------
    flux : array
        flux that will be perturbed
    age : array
        array containing age in million years
    age_min : float
        starting point of perturbation
    age_max : float
        stopping point of perturbation, anything beyond this will be constant
    factors: float
        multpying factor to perturbate the flux
    
    Returns
    -------
    flux : array
        array of flux that has been perturbed
    """
    # Array index to start and stop
    start = np.where(np.round(age)==age_min)[0][0]
    stop = np.where(np.round(age)==age_max)[0][0]

    ratio[start:stop] = np.linspace(ratio[start], ratio[start]+factor, stop-start)
    ratio[stop:] = ratio[start]+factor

    return ratio

def run_sim_config(parameter):
    """
    function to run a deterministic box model with a single dictionary as input

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

    tmin = param['tmin']
    tmax = param['tmax']
    nt = param['nt']
    dt = (tmin-tmax)*1e6 / nt
    age = np.linspace(tmin, tmax, nt)

    # Riverine flux
    Jriv = np.ones(nt)*param['Jriv']
    # Riverine isotopic ratio
    Rriv = np.ones(nt)*param['Rriv']
    # Hydrothermal flux
    Jh = np.ones(nt)*param['Jh']
    # Hydrothermal isotopic ratio
    Rh = np.ones(nt)*param['Rh']
    # Reservoir size
    N = np.ones(nt)*param['N']
    # Array to store isotopic ratio of seawater
    Rsw = np.zeros(nt)

    # Perturbation
    Jh = perturb(Jh, age, 461, 455, param['pf'])

    # Simulate
    Rsw = run_sim(nt, dt, age, Jriv, Rriv, Rsw, Jh, Rh, N)

    return {
            'age' : age,
            'Jriv':Jriv,
            'Rriv': Rriv,
            'Jh': Jh,
            'Rh' : Rh,
            'Rsw': Rsw
            }

def run_sim_config_mc(parameter, targetjson, tolerance=2e4, verbose=False):
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

    with open(targetjson) as k:
        target = json.load(k)

    tmin = param['tmin']
    tmax = param['tmax']
    nt = param['nt']
    dt = (tmin-tmax)*1e6 / nt
    age = np.linspace(tmin, tmax, nt)

    pert1_start = param['pf_age1'][0]
    pert1_stop = param['pf_age1'][1]
    pert2_start = param['pf_age2'][0]
    pert2_stop = param['pf_age2'][1]

    s = param['sampling']

    # Generate range of forcing values to sample
    Jriv_range = np.random.uniform(param['Jriv'][0], param['Jriv'][1], s)
    Rriv_range = np.random.uniform(param['Rriv'][0], param['Rriv'][1], s)
    Jh_range = np.random.uniform(param['Jh'][0], param['Jh'][1], s)
    Rh_range = np.random.uniform(param['Rh'][0], param['Rh'][1], s)
    N_range = np.random.uniform(param['N'][0], param['N'][1], s)
    pf_riv_range1 = np.random.uniform(param['pf_riv1'][0], param['pf_riv1'][1], s)
    pf_h_range1 = np.random.uniform(param['pf_h1'][0], param['pf_h1'][1], s)
    pf_riv_range2 = np.random.uniform(param['pf_riv2'][0], param['pf_riv2'][1], s)
    pf_h_range2 = np.random.uniform(param['pf_h2'][0], param['pf_h2'][1], s)

    # Resample target array
    resample = interp1d(target['age'], target['sr'])
    target_sr = resample(age)

    # Empty list to store results
    Jriv_res = []
    Rriv_res = []
    Jh_res = []
    Rh_res = []
    N_res = []
    pf_riv1_res = []
    pf_h1_res = []
    pf_riv2_res = []
    pf_h2_res = []

    # Store Rsw that match the target
    Rsw_match = np.copy(age)

    # Loop over possible forcing values
    for i in range(s):

        if verbose:
            print("run %d of %d" % (i,s))

        # Riverine flux
        Jriv = np.ones(nt)*Jriv_range[i]
        # Perturb Riverine FLux
        pf_riv1 = pf_riv_range1[i]
        Jriv = perturb(Jriv, age, pert1_start, pert1_stop, pf_riv1)
        pf_riv2 = pf_riv_range2[i]
        Jriv = perturb(Jriv, age, pert2_start, pert2_stop, pf_riv2)

        # Hydrothermal flux
        Jh = np.ones(nt)*Jh_range[i]
        # Perturb hydrothermal flux
        pf_h1 = pf_h_range1[i]
        Jh = perturb(Jh, age, pert1_start, pert1_stop, pf_h1)
        pf_h2 = pf_h_range2[i]
        Jh = perturb(Jh, age, pert2_start, pert2_stop, pf_h2)

        # Riverine isotopic ratio
        Rriv = np.ones(nt)*Rriv_range[i]

        # Hydrothermal isotopic ratio
        Rh = np.ones(nt)*Rh_range[i]


        # Reservoir size
        N = np.ones(nt)*N_range[i
                                ]
        # Array to store isotopic ratio of seawater
        Rsw = np.zeros(nt)

        # Simulate
        Rsw = run_sim(nt, dt, age, Jriv, Rriv, Rsw, Jh, Rh, N)

        # Store results if match target
        if np.mean(np.abs(Rsw-target_sr)) < tolerance:

            Rsw_match = np.vstack((Rsw_match, Rsw))

            Jriv_res.append(Jriv_range[i])
            Rriv_res.append(Rriv_range[i])
            Jh_res.append(Jh_range[i])
            Rh_res.append(Rh_range[i])
            N_res.append(N_range[i])
            pf_riv1_res.append(pf_riv1)
            pf_h1_res.append(pf_h1)
            pf_riv2_res.append(pf_riv2)
            pf_h2_res.append(pf_h2)

        else:
            pass

        # Store results as dict
    results = {
        'Jriv':np.array(Jriv_res),
        'Rriv':np.array(Rriv_res),
        'Jh':np.array(Jh_res),
        'Rh':np.array(Rh_res),
        'pf_riv1':np.array(pf_riv1_res),
        'pf_h1':np.array(pf_h1_res),
        'pf_riv2':np.array(pf_riv2_res),
        'pf_h2':np.array(pf_h2_res),
        'Rsw' : Rsw_match,
        'target':target_sr,
        'age':age
                    }


    return results

def run_sim_config_hydrothermal(parameter, targetjson, ht_flux, ht_age,
                                tolerance=2e4, verbose=False):
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

    with open(targetjson) as k:
        target = json.load(k)

    tmin = param['tmin']
    tmax = param['tmax']
    nt = param['nt']
    dt = (tmin-tmax)*1e6 / nt
    age = np.linspace(tmin, tmax, nt)

    pert1_start = param['pf_age1'][0]
    pert1_stop = param['pf_age1'][1]
    pert2_start = param['pf_age2'][0]
    pert2_stop = param['pf_age2'][1]

    s = param['sampling']

    # Generate range of forcing values to sample
    Jriv_range = np.random.uniform(param['Jriv'][0], param['Jriv'][1], s)
    Rriv_range = np.random.uniform(param['Rriv'][0], param['Rriv'][1], s)

    Rh_range = np.random.uniform(param['Rh'][0], param['Rh'][1], s)
    N_range = np.random.uniform(param['N'][0], param['N'][1], s)

    pf_riv_range1 = np.random.uniform(param['pf_riv1'][0], param['pf_riv1'][1], s)
    pf_riv_range2 = np.random.uniform(param['pf_riv2'][0], param['pf_riv2'][1], s)

    pRiv_range1 = np.random.uniform(param['pf_age1'][0], param['pf_age1'][1], s)
    pRiv_range2 = np.random.uniform(param['pf_age2'][0], param['pf_age2'][1], s)


    # Resample target array
    f = interp1d(target['age'], target['sr'])
    target_sr = f(age)

    # Resample hydrothermal flux
    f = interp1d(ht_age, ht_flux)
    Jh = f(age)

    # Empty list to store results
    Jriv_res = []
    Rriv_res = []
    Rh_res = []
    N_res = []
    pf_riv1_res = []
    pf_riv2_res = []
    priv1_res = []
    priv2_res = []

    # Store Rsw that match the target
    Rsw_match = np.copy(age)

    # Loop over possible forcing values
    for i in range(s):

        if verbose:
            print("run %d of %d" % (i,s))

        # Riverine flux
        Jriv = np.ones(nt)*Jriv_range[i]
        # Perturb Riverine FLux
        pf_riv1 = pf_riv_range1[i]
        Jriv = perturb(Jriv, age, pert1_start, pert1_stop, pf_riv1)
        pf_riv2 = pf_riv_range2[i]
        Jriv = perturb(Jriv, age, pert2_start, pert2_stop, pf_riv2)

        # Riverine isotopic ratio
        Rriv = np.ones(nt)*Rriv_range[i]
        # Riverine Ratio
        pRiv1 = pRiv_range1[i]
        Rriv = perturb_ratio(Rriv, age, pert1_start, pert1_stop, pRiv1)

        pRiv2 = pRiv_range2[i]
        Rriv = perturb_ratio(Rriv, age, pert2_start, pert2_stop, pRiv2)

        # Hydrothermal isotopic ratio
        Rh = np.ones(nt)*Rh_range[i]


        # Reservoir size
        N = np.ones(nt)*N_range[i
                                ]
        # Array to store isotopic ratio of seawater
        Rsw = np.zeros(nt)

        # Simulate
        Rsw = run_sim(nt, dt, age, Jriv, Rriv, Rsw, Jh, Rh, N)

        # Store results if match target
        if np.mean(np.abs(Rsw-target_sr)) < tolerance:

            Rsw_match = np.vstack((Rsw_match, Rsw))

            Jriv_res.append(Jriv_range[i])
            Rriv_res.append(Rriv_range[i])
            Rh_res.append(Rh_range[i])
            N_res.append(N_range[i])
            pf_riv1_res.append(pf_riv1)
            pf_riv2_res.append(pf_riv2)
            priv1_res.append(pRiv1)
            priv2_res.append(pRiv2)

        else:
            pass

        # Store results as dict
    results = {
        'Jriv':np.array(Jriv_res),
        'Rriv':np.array(Rriv_res),
        'Jh':Jh,
        'Rh':np.array(Rh_res),
        'N':np.array(N_res),
        'pf_riv1':np.array(pf_riv1_res),
        'pf_riv2':np.array(pf_riv2_res),
        'pRiv1':np.array(priv1_res),
        'pRiv2':np.array(priv2_res),
        'Rsw' : Rsw_match
                    }


    return results

def run_sim_config_riverine(parameter, targetjson, river_flux, river_age,
                            tolerance=2e4, verbose=False):
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

    with open(targetjson) as k:
        target = json.load(k)

    tmin = param['tmin']
    tmax = param['tmax']
    nt = param['nt']
    dt = (tmin-tmax)*1e6 / nt
    age = np.linspace(tmin, tmax, nt)

    pert1_start = param['pf_age1'][0]
    pert1_stop = param['pf_age1'][1]
    pert2_start = param['pf_age2'][0]
    pert2_stop = param['pf_age2'][1]

    s = param['sampling']

    # Generate range of forcing values to sample
    Rriv_range = np.random.uniform(param['Rriv'][0], param['Rriv'][1], s)

    Jh_range = np.random.uniform(param['Jh'][0], param['Jh'][1], s)
    Rh_range = np.random.uniform(param['Rh'][0], param['Rh'][1], s)

    N_range = np.random.uniform(param['N'][0], param['N'][1], s)
    pf_h_range1 = np.random.uniform(param['pf_h1'][0], param['pf_h1'][1], s)
    pf_h_range2 = np.random.uniform(param['pf_h2'][0], param['pf_h2'][1], s)

    # Resample target array
    resample = interp1d(target['age'], target['sr'])
    target_sr = resample(age)

    # Resample riverine flux
    f = interp1d(river_age, river_flux)
    Jriv = f(age)
    # Empty list to store results
    Rriv_res = []
    Jh_res = []
    Rh_res = []
    N_res = []
    pf_h1_res = []
    pf_h2_res = []

    # Store Rsw that match the target
    Rsw_match = np.copy(age)

    # Loop over possible forcing values
    for i in range(s):

        if verbose:
            print("run %d of %d" % (i,s))


        # Hydrothermal flux
        Jh = np.ones(nt)*Jh_range[i]
        # Perturb hydrothermal flux
        pf_h1 = pf_h_range1[i]
        Jh = perturb(Jh, age, pert1_start, pert1_stop, pf_h1)
        pf_h2 = pf_h_range2[i]
        Jh = perturb(Jh, age, pert2_start, pert2_stop, pf_h2)

        # Riverine isotopic ratio
        Rriv = np.ones(nt)*Rriv_range[i]

        # Hydrothermal isotopic ratio
        Rh = np.ones(nt)*Rh_range[i]


        # Reservoir size
        N = np.ones(nt)*N_range[i]

        # Array to store isotopic ratio of seawater
        Rsw = np.zeros(nt)

        # Simulate
        Rsw = run_sim(nt, dt, age, Jriv, Rriv, Rsw, Jh, Rh, N)

        # Store results if match target
        if np.mean(np.abs(Rsw-target_sr)) < tolerance:

            Rsw_match = np.vstack((Rsw_match, Rsw))

            Rriv_res.append(Rriv_range[i])
            Jh_res.append(Jh_range[i])
            Rh_res.append(Rh_range[i])
            N_res.append(N_range[i])
            pf_h1_res.append(pf_h1)
            pf_h2_res.append(pf_h2)

        else:
            pass

        # Store results as dict
    results = {
        'Rriv':np.array(Rriv_res),
        'Jh':np.array(Jh_res),
        'Rh':np.array(Rh_res),
        'pf_h1':np.array(pf_h1_res),
        'pf_h2':np.array(pf_h2_res),
        'Rsw' : Rsw_match
        }


    return results

def run_sim_steady_state(parameter, target_array, tolerance=2e-4,
                         mode='random',
                         riverine_flux=[],
                         riverine_age=[],
                         hydrothermal_flux=[],
                         hydrothermal_age=[]):
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

    if mode=='random':
        # Generate range of forcing values to sample
        Jriv_range = np.random.uniform(param['Jriv'][0], param['Jriv'][1],s)
        Rriv_range = np.random.uniform(param['Rriv'][0], param['Rriv'][1],s)
        Jh_range = np.random.uniform(param['Jh'][0], param['Jh'][1], s)
        Rh_range = np.random.uniform(param['Rh'][0], param['Rh'][1], s)


        # Empty list to store results
        Jriv_res = np.zeros((nt,s))
        Rriv_res = np.zeros((nt,s))
        Jh_res = np.zeros((nt,s))
        Rh_res = np.zeros((nt,s))
    
        for j in tqdm(range(nt), desc='random MC progress', ncols=80):
        # Loop over possible forcing values
            for i in range(s):
                # Riverine flux
                Jriv = Jriv_range[i]
        
                # Hydrothermal flux
                Jh = Jh_range[i]
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
    
    
        print('MC simulation with randomized parameters is done')
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

        # Generate range of forcing values to sample
        Rriv_range = np.random.uniform(param['Rriv'][0], param['Rriv'][1],s)
        Jh_range = np.random.uniform(param['Jh'][0], param['Jh'][1], s)
        Rh_range = np.random.uniform(param['Rh'][0], param['Rh'][1], s)
    
    
        # Empty list to store results
        Jriv_res = np.zeros((nt,s))
        Rriv_res = np.zeros((nt,s))
        Jh_res = np.zeros((nt,s))
        Rh_res = np.zeros((nt,s))
    
        for j in tqdm(range(nt), desc='riverine MC progress', ncols=80):
        # Loop over possible forcing values
            for i in range(s):
                # Riverine flux
                Jriv = Jriv_input[j]
        
                # Hydrothermal flux
                Jh = Jh_range[i]
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
    
        for j in tqdm(range(nt), desc='hydrothermal MC progress'):
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
    
        for j in tqdm(range(nt), desc='multi-input MC progress', ncols=80):
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

if __name__ =='__main__':
    import time
    starttime = time.time()
    par = run_sim_config_mc('param_mc.json', 'target.json', tolerance=5e-5, verbose=True)
    exectime = time.time() - starttime
    print('Execution time: %.2f s'%exectime)
    for i in range(len(par['Rsw'])-1):
        plt.plot(par['Rsw'][0], par['Rsw'][i+1], 'k--', alpha=0.3)
    plt.plot(par['age'], par['target'], 'r-')
    plt.plot(par['age'], par['target']+1e-4, 'r--')
    plt.plot(par['age'], par['target']-1e-4, 'r--')
    plt.show()

