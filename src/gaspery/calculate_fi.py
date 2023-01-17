import numpy as np 
import scipy
import pandas as pd
import random
import exoplanet
import astropy 
import pymc3
import pymc3_ext
import celerite2
from tinygp import kernels, GaussianProcess
from numpy.linalg import inv, det, solve, cond
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap


class Star(object): 
    """
    Star objects contain the target star's properties, to be fed into the cov_matrix_jax() function for calculating the covariance matrix.

    Attributes: 
    - sigma_wn_rv: white/photon noise amplitude [cm/s]; free param taken from Langellier et al 2020
    - Tau: active region lifetime [days]; free param taken from Langellier et al 2020
    - eta: smoothing parameter; free param taken from Langellier et al 2020
    - Prot: rotation period [days]; free param taken from Langellier et al 2020
    - sigma_qp_rv: correlated noise amplitude [cm/s]; free param taken from Langellier et al 2020

    """

    def __init__(
        self, sigma_wn_rv, Tau=None, eta=None, Prot=None, sigma_qp_rv=None, **kwargs 
    ):
        self.sigma_wn_rv = sigma_wn_rv
        try:
            self.Tau = Tau
            self.eta = eta
            self.Prot = Prot
            self.sigma_qp_rv = sigma_qp_rv
        except:
            pass


    def param_list(self):
        """
        Make list of params based on Star object attributes to feed into cov_matrix_jax()

        Input:
        - Star object with attributes from __init__()
        Return:
        - Params: list of parameters to feed into cov_matrix_jax(), for either white or correlated noise case

        """

        # if correlated noise 
        try:
            params = [self.sigma_wn_rv, self.Tau, self.eta, self.Prot, self.sigma_qp_rv]
        # elif white noise only
        except:
            params = [self.sigma_wn_rv]

        return params


    def cov_matrix_general(self, t, kernel):
        """
        Build a covariance matrix that will be used in the calculation of the Fisher Information Matrix, using a generalized GP kernel.
        
        Inputs:
        - self: Star object
        - t: time series of length N observations; np.array [day]
        - kernel: kernel object from tinygp.kernels.Kernel class

        Outputs:
        - N by N matrix of covariance elements following Equations 1 & 2 of Langellier et al 2020
        
        """

        # if we are in correlated noise regime, using a quasi-periodic Gaussian Process kernel
        if all([self.sigma_wn_rv, self.Tau, self.eta, self.Prot, self.sigma_qp_rv]):
            sigma_wn_rv, Tau, eta, Prot, sigma_qp_rv = self.sigma_wn_rv, self.Tau, self.eta, self.Prot, self.sigma_qp_rv

            # create N by N matrix, where N is length of observations time series
            k = jnp.zeros((len(t),len(t)))
            #print("strat: ", t)
            for i in range(len(t)):
                for j in range(len(t)):
                    term1 = jnp.exp(((t[i]-t[j])**2)/(2*Tau**2))
                    term2 = kernel.evaluate(t[i], t[j])
                    k = term1 * term2
            
            K = sigma_qp_rv**2 * k + sigma_wn_rv**2 * jnp.diag(np.ones(len(t)))
            #print("shape: ", K.shape)
            
            return K

        # if we are in white noise regime
        elif all([self.sigma_wn_rv]): 
            sigma_wn_rv = self.sigma_wn_rv**2 
            sigma = np.diag(sigma_wn_rv * np.ones(len(t)))

            return sigma

        # else, we are not in a valid regime
        else:
            raise AttributeError('Please input at least sigma_wn_rv for photon noise.')


    def cov_matrix(self, t):
        """
        Build a covariance matrix that will be used in the calculation of the Fisher Information Matrix, using a quasi-periodic GP kernel.
        
        Inputs:
        - self: Star object
        - t: time series of length N observations; np.array [day]

        Outputs:
        - N by N matrix of covariance elements following Equations 1 & 2 of Langellier et al 2020
        
        """
            
        # if we are in correlated noise regime, using a quasi-periodic Gaussian Process kernel
        if all([self.sigma_wn_rv, self.Tau, self.eta, self.Prot, self.sigma_qp_rv]):
            sigma_wn_rv, Tau, eta, Prot, sigma_qp_rv = self.sigma_wn_rv, self.Tau, self.eta, self.Prot, self.sigma_qp_rv

            # create N by N matrix, where N is length of observations time series
            k = jnp.zeros((len(t),len(t)))
            #print("strat: ", t)
            for i in range(len(t)):
                for j in range(len(t)):
                    term1 = ((t[i]-t[j])**2)/(2*Tau**2)
                    term2 = (1/(2*eta**2)) * (jnp.sin(jnp.pi * (t[i] - t[j])/Prot))**2
                    arg = -term1 - term2
                    #k[i][j] = jnp.exp(arg)
                    k = jnp.exp(arg)
                    #k = k.at[i]
                    #x = x.at[idx].set(y)
            
            K = sigma_qp_rv**2 * k + sigma_wn_rv**2 * jnp.diag(np.ones(len(t)))
            #print("shape: ", K.shape)
            
            return K

        # if we are in white noise regime
        elif all([self.sigma_wn_rv]): 
            sigma_wn_rv = self.sigma_wn_rv**2 
            sigma = np.diag(sigma_wn_rv * np.ones(len(t)))

            return sigma

        # else, we are not in a valid regime
        else:
            raise AttributeError('Please input at least sigma_wn_rv for photon noise.')


class Planets(object):
    """
    Planet objects contain the target system's planet properties, to be fed into the clam_jax_fim() function for calculating Fisher Information.
    Multiple planets are allowed.

    Attributes:
    - K: RV semi-amplitude [cm/s]
    - p: planet orbital period [days]
    - T0: planet central transit time [BJD]

    """ 

    def __init__(
        self, K, p, T0, **kwargs
    ):
        self.K = K
        self.p = p
        self.T0 = T0


    def theta_list(self):
        """
        Make list of params based on Planets object attributes to feed into Fisher Info calculation machinery

        Input:
        - Planets object with attributes from __init__()
        Return:
        - Theta: list of parameters to feed into clam_jax_fim()

        """

        theta = [self.K, self.p, self.T0]

        return theta


def cov_matrix_jax(t, sigma_wn_rv, params=[]):
    """
    Build a covariance matrix, Sigma, that will be used in the calculation of the Fisher Information Matrix.
    Deprecated bc incompatibility with object-oriented framework.
    
    Inputs:
    - t: time series of length N observations; np.array [day]
    - sigma_wn_rv: white noise amplitude; free param taken from Langellier et al 2020
    - params: list of stellar correlated noise parameters
        - Tau: active region lifetime [days]; free param taken from Langellier et al 2020
        - eta: smoothing parameter; free param taken from Langellier et al 2020
        - Prot: rotation period [days]; free param taken from Langellier et al 2020
        - sigma_qp_rv: correlated noise amplitude; free param taken from Langellier et al 2020
    
    Outputs:
    - N by N matrix of covariance elements following Equations 1 & 2 of Langellier et al 2020
    
    """
        
    # if we are in correlated noise regime, using a quasi-periodic kernel
    if len(params) != 0:
        Tau, eta, Prot, sigma_qp_rv = params[0], params[1], params[2], params[3]

        # create N by N matrix, where N is length of observations time series
        k = jnp.zeros((len(t),len(t)))
        
        for i in range(len(t)):
            for j in range(len(t)):
                term1 = ((t[i]-t[j])**2)/(2*Tau**2)
                term2 = (1/(2*eta**2)) * (jnp.sin(jnp.pi * (t[i] - t[j])/Prot))**2
                arg = -term1 - term2
                #k[i][j] = jnp.exp(arg)
                k = jnp.exp(arg)
                #k = k.at[i]
                #x = x.at[idx].set(y)
        
        K = sigma_qp_rv**2 * k + sigma_wn_rv**2 * jnp.diag(np.ones(len(t)))
        #print("shape: ", K.shape)
        
        return K

    # else, we are in white noise regime
    elif len(params) == 0: 
        sigma_wn_rv = sigma_wn_rv**2 
        sigma = np.diag(sigma_wn_rv * np.ones(len(t)))

        return sigma


@jax.jit
def clam_jax_fim(t, sigma, theta): 
    """
    Calculate the generalized Fisher Information Matrix using JAX Jacobian.
    Generalizable to arbitrary parameters.
    Now, with second term for correlated noise. 
    
    Inputs: 
    - t: time series of length N observations; np.array [day]
    - sigma: RV measurement uncertainties associated with each observation; np.array of length N [cm/s]
    - theta: planet orbital parameters; eg. [K, P, T0]; np.array
        - K: RV semi-amplitude [cm/s]
        - P: planet period [days]
        - T0: mean transit time [day]
    
    Output:
    - Fisher Information Matrix: len(theta)xlen(theta) matrix; np.array
    
    """
    
    def inner(params):
        return model_jax(t, params[0], params[1], params[2])
    
    sigma += 1e-6 # add jitter
    factor = jnp.linalg.solve(sigma, jnp.identity(len(sigma))) # take inverse of covariance matrix
    #factor = jnp.linalg.inv(sigma)
    #factor = jnp.where(sigma > 0, sigma_inv, 0)  
    
    J = jax.jacobian(inner)(theta)
    return J.T @ factor @ J


def model_jax(t, K, P, T0): 
            """
            Radial velocity model, given timestamps and planetary orbital parameters, but JAXified
            (basically, np --> jnp)
            
            Inputs: 
            - t: time series of length N observations; np.array [day]
            - K: RV semi-amplitude [cm/s]
            - P: planet period [days]
            - T0: mean transit time [day]

            """
            
            arg = (2*jnp.pi/P)*(t-T0)
            rv = -K * jnp.sin(arg)
            
            return rv