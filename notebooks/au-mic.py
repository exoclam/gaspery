import numpy as np 
import scipy
print(np.__version__)
print(scipy.__version__)
import tqdm
import pandas as pd
import random
import astropy 
from numpy.linalg import inv, det, solve, cond
from tqdm import tqdm
from astropy.time import Time

import matplotlib.pyplot as plt
import matplotlib

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
#from jax import random

from gaspery import calculate_fi, strategies, utils
from tinygp import kernels, GaussianProcess

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)

path = '/Users/chris/Desktop/gaspery/'
path = '/blue/sarahballard/c.lam/gaspery/'

random_generator = np.random.default_rng(seed=42)

# correlated noise parameters, from Klein+ 2021 for AU Mic
Prot = 4.86 # days
Tau = 100/np.sqrt(2) # active region lifetime; days
eta = 0.4/np.sqrt(2) # 0.1, 0.3, 0.58, 0.9 # smoothing parameter
sigma_qp_rv = 47 # modified Jeffreys prior +11, -8 [m/s]
sigma_wn_rv = 5 # [m/s]

# planet parameters
K = 8.5 # m/s
K_err = 2.25 # m/s (average of +2.3, -2.2 m/s)
p = 8.46 # days
T0 = 2458651.993 # central transit time, in BJD, on 19 June 2019

n_obs = 30 

### choose start time as date of this writing
start = '2023-03-01T10:00:00'
start = Time(start, format='isot', scale='utc').jd

dropout = 0.1
offs = [[start+14, start+28], [start+42, start+56], [start+70, start+84]]

### the parameter I'm varying
params = [Tau, eta, Prot, sigma_qp_rv, sigma_wn_rv]
theta = [K, p, T0]

dim = 98 # 128
n_observations = np.linspace(4, 100, 97).astype(int)
cadences1 = np.linspace(0.30, 3.4976378, int(dim/4))
cadences2 = np.linspace(3.4976378, 5.04488189, int(dim/2))[1:]
cadences3 = np.linspace(5.04488189, 6.85, int(dim/2))[1:]
cadences = np.concatenate([cadences1, cadences2, cadences3])

def compute_sensitivity_map_residual():

    """
    Wrapper for computing P_orb vs cadence sensitivity maps for Figure 2 in paper
    """

    sigma_ks = np.ones(len(n_observations)*len(cadences)).reshape((len(n_observations),len(cadences)))
    fi_ks = np.ones(len(n_observations)*len(cadences)).reshape((len(n_observations),len(cadences)))

    for enum_o, o in enumerate(tqdm(n_observations)):
        for enum_c, c in enumerate(cadences):
                    
            try:
                sigma_ks_temp = []
                for i in range(10):
                    # randomize start date/time, so that we are not subject to accidentally falling on an uninformative phase
                    start_random = random_generator.uniform(start, start+p) 
                    
                    # instantiate Strategy object in order to build time series of observations
                    strategy = strategies.Strategy(n_obs = o, start = start_random, offs=[], dropout=0.)

                    # build strategy aka time series of observations
                    strat = strategy.gappy(c)

                    ### calculate FI and sigma_K
                    # build covariance matrix, characterized by a correlated noise model of the stellar signal
                    kernel = kernels.ExpSineSquared(scale=Prot, gamma=1/(2*eta**2)) # first term of exponential
                    kernel *= kernels.ExpSquared(scale=Tau) # other term of exponential
                    kernel *= sigma_qp_rv**2 # multiply by scalar

                    # instantiate gaspery Star object in order to feed covariance matrix with white/correlated noise
                    star = calculate_fi.Star(sigma_wn_rv = sigma_wn_rv, Tau = Tau, eta = eta, Prot = Prot, sigma_qp_rv = sigma_qp_rv)
                    sigma_correlated = star.cov_matrix_general(strat, kernel)
                    sigma_white = np.diag(sigma_wn_rv*np.ones(len(strat))) 
                    sigma_correlated += 1e-6 
                    sigma_white += 1e-6 

                    # populate arguments for Fisher Info calculator
                    args_correlated = np.array(strat), sigma_correlated, jnp.array(theta, dtype=float)
                    args_white = np.array(strat), sigma_white, jnp.array(theta, dtype=float)

                    # calculate FI
                    fim_correlated = calculate_fi.clam_jax_fim(*args_correlated).block_until_ready()
                    fim_white = calculate_fi.clam_jax_fim(*args_white).block_until_ready()

                    # calculate sigma_K
                    inv_fim_correlated = inv(fim_correlated)
                    sigma_k_correlated = np.sqrt(inv_fim_correlated)[0][0]

                    inv_fim_white = inv(fim_white)
                    sigma_k_white = np.sqrt(inv_fim_white)[0][0]

                    # for 2D plots and testing
                    #sigma_ks[enum_o][enum_c] = sigma_k
                    ###sigma_ks_stable[enum1][enum2] = sigma_k_stable
                    ###sigma_ks_solve[enum1][enum2] = sigma_k_solve   
                    sigma_ks_temp.append(np.abs(sigma_k_correlated - sigma_k_white))
                    
            except Exception as e:
                print(e, o, c)
                sigma_ks[enum_o][enum_c] = np.nan

            sigma_k_all = np.nanmean(sigma_ks_temp)
            sigma_ks[enum_o][enum_c] = sigma_k_all

    return sigma_ks

sigma_ks_residual = compute_sensitivity_map_residual()
print(sigma_ks_residual)
print(np.nanmax(sigma_ks_residual), np.nanmin(sigma_ks_residual))
"""
x, y = np.meshgrid(cadences, n_observations)
fig = plt.pcolormesh(y,x,sigma_ks, norm=matplotlib.colors.LogNorm())

cbar = plt.colorbar(fig)
cbar.ax.set_ylabel(r"$\sigma_k$[m/s]")
plt.xlabel("number of observations")
plt.ylabel("time between observations [day]")
plt.title(f"correlated noise") 
plt.savefig(path + f"plots/n_obs-vs-cadence-correlated-meters-averaged.png", format="png")
plt.show()
"""

"""
I want to see what the difference between white and correlated noise is, for each case on these plots. 
"""
x, y = np.meshgrid(cadences, n_observations)
fig = plt.pcolormesh(y,x,sigma_ks_residual, norm=matplotlib.colors.LogNorm())

cbar = plt.colorbar(fig)
cbar.ax.set_ylabel(r"$\sigma_k$[m/s]")
plt.xlabel("number of observations")
plt.ylabel("time between observations [day]")
plt.title(f"correlated noise") 
plt.savefig(path + f"plots/n_obs-vs-cadence-correlated-vs-white-meters-averaged.png", format="png")
plt.show()