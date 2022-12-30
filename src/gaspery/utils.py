import numpy as np 
import scipy
import pandas as pd
import random
import exoplanet
import astropy 
import pymc3
import pymc3_ext
import celerite2
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