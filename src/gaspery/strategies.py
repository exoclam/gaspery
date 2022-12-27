import numpy as np 
import scipy
import tqdm
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

class Strategy(n_obs, cadence, start, offs=[], dropout=0.):
    """

    Functions that make strategies, based on user inputs. 

    Inputs:
    - n_obs: number of observations to make; this is a fixed goal [int]
    - cadence: amount of time between observations [days]
    - start: start time [BJD]
    - offs: all pairs of off-night spans, described by their start/end times [list of two-element lists of floats]
    - dropout: percent of dropout due to weather and other uniformly distributed unfortunate events [float]

    Output:
    - Strategy object, on which the following strategy-building functions can operate.

    """

    def remove(l,n):
        """
        For the dropout function, randomly remove some fraction of elements.

        Inputs: 
        - l: observation timestamps [np.array]
        - n: fraction to remove [float]

        Output: observation timestamps with random elements removed [np.array]
        """

        return np.sort(random.sample(list(l),int(len(l)*(1-n))))


    def gappy(self):
        """
        Specify how many nights in a row to observe and how many nights in a row to skip,
        eg. input: {nights on, cadence, percentage of completed observations} 
        
        Input: 
        - Strategy object
        
        Output:
        - observation times: ndarray of floats
        
        """
        
        total_t = []
        start_temp = start
        n_tot = 0
        n = 0
        
        if len(offs) > 0:
            # keep making observations until you've reached your prescribed budget of observations
            while n_tot < n_obs:
                # allow for multiple spans of off nights
                for off in offs:

                    end_temp = off[0]

                    # rearrange from end = start + n_obs * cadence to get on chunk
                    n = int(np.ceil((end_temp - start_temp)/cadence))

                    try: # if off != offs[-1]: # if there are yet more offs? 
                        t = np.linspace(start_temp, end_temp, n, endpoint=False)
                        # add jitter ~ N(0, 1 hr) to timestamps
                        t += np.random.normal(0, 1./24)
                        total_t = np.concatenate([total_t, t])
                        n_remaining = n_obs - len(total_t)

                    except: # if there aren't, then we just straight shot til n_obs is fulfilled
                        #print("exception")
                        end_temp = start_temp + n_remaining * cadence
                        t = np.linspace(start_temp, end_temp, n_remaining, endpoint=False)

                        # add jitter ~ N(0, 1 hr) to timestamps
                        t += np.random.normal(0, 1./24)

                        total_t = np.concatenate([total_t, t])


                    # new start time
                    start_temp = off[1]

                    n_tot = len(total_t)

                    if n_tot >= n_obs:
                        break

                # let's say there's no more offs and yet more observations to make
                # Then, we just straight shot til n_obs is fulfilled          

            # pare down total_t if the last concatenation overshoots n_obs
            # I have logic for dealing with that if I get through all off nights
            # But n_obs gets hit before then, this is to take care of that.
            n_extra = n_tot - n_obs
            if n_extra > 0:
                total_t = total_t[:-n_extra]
        
        else:
            total_t = make_t(n_obs, cadence, start)
        
        # dropout some observations based on dropout
        total_t = remove(total_t, dropout)
        
        return total_t

    
    def on_vs_baseline_balanced(start, n_obs, on, baseline, perfect_flag=False):
        """
        Construct observing strategy given on nights, baseline, and n_obs.
        Distribute on nights across time series while meeting the prescription.
        
        Inputs: 
        - start: start date [BJD]
        - n_obs: number of observations budgeted for
        - on: number of consecutive nights of observation [days]
        - baseline: overall observing window [days]
        - perfect_flag: if True, only make strategies for combos that have equal numbers of offs in each block
        if False, make strategies for any combo that has off periods = on periods - 1
        
        Returns: 
        - strat: time series of dates of observations [list of floats]
        
        """
        
        temp_start = start
        
        n_on_periods = int(np.ceil(n_obs/on))
        
        # skip combinations that don't necessitate a perfectly balanced strat
        if perfect_flag == True:    
            if (baseline - n_obs) % (n_on_periods - 1) != 0: 
                strat = []
                return strat
    
        n_mixed_periods = int(n_on_periods - 1)
        
        block_length = int(np.floor((baseline - n_obs)/n_mixed_periods))
        
        off_length = block_length - on
        
        on_total = int(n_mixed_periods * on)

        nights_total = int(n_mixed_periods * block_length)
        
        on_residual = n_obs - on_total

        nights_residual = baseline - nights_total
        

        ### construct 'on-masks'
        ons = []
        for i in range(n_mixed_periods):
            on_i = np.arange(temp_start, temp_start + on, 1)
            ons.append(on_i)
                            
            temp_start += on
            
            # minimum off block length
            off1 = np.floor((baseline - n_obs)/n_mixed_periods)
            
            # number of off blocks with an extra night
            off2 = (baseline - n_obs) % n_mixed_periods
            
            if i+1 <= off2:
                temp_start += (off1+1)
            else: 
                temp_start += off1
                            
        ### last on block to top off
        ons.append(np.arange(temp_start, temp_start + on_residual, 1))
        
        ### flatten list
        ons = [item for sublist in ons for item in sublist]
        
        return ons


