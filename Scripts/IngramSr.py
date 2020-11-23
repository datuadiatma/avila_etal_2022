# Import modules
import numpy as np
import matplotlib.pyplot as plt

# Define Governing equations as a function
def drodt(jr, rr, ro, jh, rh, n):
	Sr = (jr * (rr - ro) + jh * (rh - ro)) / n
	return Sr

# Input parameters (baseline values)
N = 1.25e17    # Sr ocean reservoir size in mole
Jr = 2.2e10    # Sr riverine flux in mole/yr
Rr = 0.710     # Sr riverine isotopic ratio
Jh = 1.2e10    # Sr hydrothermal flux
Rh = 0.703     # Sr hydrothermal isotopic ratio
Ro0 = (Jr*Rr + Jh*Rh)/(Jr+Jh)  #Steady state seawater Sr isotopic ratio

# Time array
tmin = 0
tmax = 1.16e7
nt = 117		# Time step to solve ODE
time = np.linspace(tmin, tmax, nt)
dt = (tmax - tmin) / nt

# Parameters vectorization
Nvec = np.ones(nt)*N
Jrvec = np.ones(nt)*Jr
Rrvec = np.ones(nt)*Rr
Jhvec = np.ones(nt)*Jh
Rhvec = np.ones(nt)*Rh
Rovec = np.ones(nt)*Ro0

# Adjust forcing
Jrvec[14:22] = np.linspace(Jr, Jr*2, 8)
Jrvec[22:] = Jr * 2

# Implement Model using first order Euler method
for i in range(nt-1):
    Rovec[i+1] = Rovec[i] + drodt(Jrvec[i], Rrvec[i], Rovec[i], Jhvec[i], Rhvec[i], Nvec[i]) * dt

# Plotting Results
plt.figure(figsize=(10,8))
plt.subplot(2,1,1)
plt.plot(time, Jrvec)
plt.title('Riverine Flux')
plt.subplot(2,1,2)
plt.plot(time, Rovec)
plt.xlabel('Sr isotopes seawater')
plt.show()