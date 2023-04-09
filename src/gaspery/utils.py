import numpy as np 
import scipy
import pandas as pd
import random
import astropy 
from numpy.linalg import inv, det, solve, cond
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

"""
Function that builds cadence vs orbital period sensitivity map
"""

"""
Function that builds observing baseline vs consecutive-on-nights sensitivity map
"""

"""
Function that overlays observations on top of RV model for visualization and troubleshooting
"""

def calculate_rv(m1, m2, P, e=0, i=np.pi/2):
    """
    Calculate radial velocity semi-amplitude, K, based on planet and stellar parameters
    Eqn 14 from Lovis & Fischer 2010 
    (http://exoplanets.astro.yale.edu/workshop/EPRV/Bibliography_files/Radial_Velocity.pdf)
    
    Inputs:
    - m1: primary (stellar) mass [Solar mass]
    - m2: secondary (planet) mass [Earth mass]
    - P: planet orbital period [days]
    - e: eccentricity; defaults to circular case e=0
    - i: inclination [radians]; defaults to i=pi/2 edge on
    
    Output: 
    - K: RV semi-amplitude [m/s]
    
    """
    
    MJup = 318 # Earth masses
    f1 = 28.4329 / np.sqrt(1 - e**2)
    f2 = m2 * np.sin(i) / MJup
    f3 = m1**(-2./3) # actually m1+m2 but m2 --> 0
    f4 = (P/365)**(-1./3)
    
    return f1 * f2 * f3 * f4
