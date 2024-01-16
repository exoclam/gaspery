from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import datetime
from numpy.linalg import inv, det, solve, cond

from gaspery import calculate_fi, strategies, utils
from tinygp import kernels, GaussianProcess

jax.config.update("jax_enable_x64", True)

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

path = '/Users/chris/Desktop/gaspery/'
path = '/blue/sarahballard/c.lam/gaspery/'

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)

import arviz as az

def mean_function(params, X):
    
    return -params['K'] * jnp.sin(2 * jnp.pi * (X - params['T0']) / params['p'])

### AU Mic b parameters from Klein+ 
K = 8.5
p = 8.46
T0 = 2458651.993 - 2458650
theta = [8.5, 8.46, T0]

mean_params = {
    "K": K,
    "p": p,
    "T0": T0}

### random seeds and initializing global variables
random = np.random.default_rng(seed=4) # formerly seed of 4
random_generator = np.random.default_rng(seed=4)
random_perturbation = np.random.default_rng(seed=8)

n_obs = 30
start = T0
offs = []

### star
sigma_wn_rv = 5.
sigma_qp_rv = 47. #145 #47
Prot = 4.86
Tau = 110. # 30, 110
eta = 0.37
yerr = sigma_wn_rv

# prepare MCMC ingredients
hyperparams = [sigma_qp_rv, Prot, Tau, eta]
theta = [K, p, T0]

### GP kernel
kernel = kernels.ExpSineSquared(scale=Prot, gamma=1/(2*eta**2)) # first term of exponential
kernel *= kernels.ExpSquared(scale=Tau) # other term of exponential
kernel *= sigma_qp_rv**2 # multiply by scalar

sigma_ks = []
sigma_ks_mcmc_minus = []
sigma_ks_mcmc_median = []
sigma_ks_mcmc_plus = []

res_trainings = []
res_injecteds = []

# randomize start date/time, so that we are not subject to accidentally falling on an uninformative phase
for i in range(10):
    ### strategy not in quadrature
    """
    start_random = random_generator.uniform(start, start+p) 
    strategy = strategies.Strategy(n_obs = n_obs, start = start_random, offs=offs, dropout=0.)
    grid_strat = np.array(strategy.on_vs_off(on=1, off=p/2 - 1, twice_flag=False)) # off=p-1 :(; off=1.115 :)
    strat = grid_strat + random_perturbation.normal(0, 1./12, len(grid_strat)) # formerly spread of 1./6 
    """

    ### strategy in quadrature
    strategy = strategies.Strategy(n_obs = n_obs, start = start+2.115, offs=offs, dropout=0.)
    grid_strat = np.array(strategy.on_vs_off(on=1, off=p/2 - 1, twice_flag=False)) 
    strat = []
    for s in grid_strat:
        # draw three random times around each location of a trough or peak, with 2 hr spread
        strat.append(random.normal(loc=s, scale=1/12, size=3))
    strat = np.array(strat).ravel()
    print("strat: ", strat)
    
    ### fine grid for the MCMC
    # x_fine must pass through strat; this way, I can make it as fine as I want
    n = 10
    x_fine = np.linspace(grid_strat[0], grid_strat[-1], int((grid_strat[-1] - grid_strat[0]) * n ) + 1)

    # build x_fine as intrinsically random; concatenate strategy to x_fine
    x_fine = np.concatenate((x_fine, strat))
    x_fine = np.sort(x_fine)

    ### generate fake ground truth and validation set
    planet_fine = calculate_fi.model_jax(x_fine, [theta[0], theta[1], theta[2]])
    gp_fine = GaussianProcess(kernel, x_fine) 
    star_fine = gp_fine.sample(jax.random.PRNGKey(4), shape=(1,)) 
    observed_fine = star_fine + planet_fine + random.normal(0, sigma_wn_rv, len(planet_fine))

    # assemble parent DataFrame, and split into training and validation sets
    df_fine = pd.DataFrame({'x': x_fine, 'y': observed_fine[0], 
                                'planet': planet_fine, 'star': star_fine[0]})
    df_fine = df_fine.drop_duplicates(subset=['x'])

    not_injected = df_fine.loc[df_fine.x.isin(strat)]

    injected_x = random.choice(x_fine, 30, replace=False)

    injected = df_fine.loc[df_fine.x.isin(injected_x)]

    ### gaspery sigma_K
    # instantiate gaspery Star object in order to feed covariance matrix with white/correlated noise
    star = calculate_fi.Star(sigma_wn_rv = sigma_wn_rv, Tau = Tau, eta = eta, 
                            Prot = Prot, sigma_qp_rv = sigma_qp_rv)

    #strat = np.array(strategy.on_vs_off(on=1, off=0, twice_flag=False))
    # calculate covariance matrix
    sigma = star.cov_matrix_general(strat, kernel)

    # populate arguments for Fisher Info calculator
    args = np.array(strat), sigma, jnp.array(theta, dtype=float)

    # calculate FI
    fim = calculate_fi.clam_jax_fim(*args).block_until_ready()

    # invert FI matrix
    inv_fim = inv(fim)

    # top left element of matrix corresponds with RV semi-amplitude, K
    sigma_k = np.sqrt(inv_fim)[0][0]
    #print("start: ", start_random, "expected value of uncertainty on K: ", sigma_k, " m/s")
    sigma_ks.append(sigma_k)


    ### MCMC sampling
    def numpyro_model(t, y, yerr, hyperparams, theta, injected_x, not_injected_x, t_fine):        
            
        sigma_qp_rv = hyperparams[0]
        Prot = hyperparams[1]
        Tau = hyperparams[2]
        eta = hyperparams[3]
        
        K = theta[0]
        p = theta[1]
        T0 = theta[2]

        # build covariance matrix, characterized by a correlated noise model of the stellar signal
        kernel = kernels.ExpSineSquared(scale=Prot, gamma=1/(2*eta**2)) # first term of exponential
        kernel *= kernels.ExpSquared(scale=Tau) # other term of exponential
        kernel *= sigma_qp_rv**2 # multiply by scalar
        
        # sample hyperparameters for planet mean model
        p = numpyro.sample("P", dist.Normal(p, 0.00004)) 
        K = numpyro.sample("K", dist.TruncatedNormal(0., 100., low=0.)) # formerly K, 2.25, but that's too informative
        #K = numpyro.sample("K", dist.Uniform(0., 100.))
        T0 = numpyro.sample("T0", dist.Normal(T0, 1))
        mean_params = {"K": K, "P": p, "T0": T0}
            
        def mean_function(t):
            """
            Mean model is the planet's Keplerian.
            """

            return -mean_params['K'] * jnp.sin(2 * jnp.pi * (t - mean_params['T0']) / mean_params['P'])
            
        gp = GaussianProcess(kernel, t, diag=yerr**2, mean=mean_function) # mean_function
        
        numpyro.sample("gp", gp.numpyro_dist(), obs=y)
        
        if y is not None:
            # condition on y; evaluate on these three different time vectors
            numpyro.deterministic("pred", gp.condition(y, t).gp.loc) # evaluate on each strat time 
            numpyro.deterministic("pred-plot", gp.condition(y, t_fine).gp.loc) # evaluate on finer strat time, for plots
            numpyro.deterministic("inj-rec", gp.condition(y, injected_x).gp.loc) # evaluate on injected times only
            
            # same but without including the mean, to separately evaluate planet and stellar fits
            numpyro.deterministic("pred-star", gp.condition(y, t, include_mean=False).gp.loc) 
            numpyro.deterministic("pred-plot-star", gp.condition(y, t_fine, include_mean=False).gp.loc) 
            numpyro.deterministic("inj-rec-star", gp.condition(y, injected_x, include_mean=False).gp.loc) 

    nuts_kernel = NUTS(numpyro_model, dense_mass=True, target_accept_prob=0.9)
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=1000, # 1000
        num_samples=8000, # 5000
        num_chains=2,
        progress_bar=True,
    )
    rng_key = jax.random.PRNGKey(34923)

    # run MCMC
    gp_output = mcmc.run(rng_key, t=np.array(not_injected.x), y=np.array(not_injected.y), yerr=sigma_wn_rv, # X, y vs t_plot, y_plot
             hyperparams=hyperparams, theta=theta, injected_x=np.array(injected.x), 
                         not_injected_x=np.array(not_injected.x), t_fine=x_fine)
    samples = mcmc.get_samples()

    preds = samples["pred"].block_until_ready()  # Blocking to get timing right
    preds_plot = samples["pred-plot"].block_until_ready()  
    preds_star = samples["pred-star"].block_until_ready()  
    preds_inj_rec = samples["inj-rec"].block_until_ready()  
    preds_plot_star = samples["pred-plot-star"].block_until_ready()  
    preds_inj_rec_star = samples["inj-rec-star"].block_until_ready()  

    # read out MCMC posteriors
    data = az.from_numpyro(mcmc)    
    #print(np.percentile(data.posterior.data_vars['K'], 50) - np.percentile(data.posterior.data_vars['K'], 16))
    sigma_ks_mcmc_plus.append(np.percentile(data.posterior.data_vars['K'], 84) - np.percentile(data.posterior.data_vars['K'], 50))
    sigma_ks_mcmc_minus.append(np.percentile(data.posterior.data_vars['K'], 50) - np.percentile(data.posterior.data_vars['K'], 16))
    sigma_ks_mcmc_median.append(np.percentile(data.posterior.data_vars['K'], 50))

    ### residuals
    q = np.percentile(preds, [16, 50, 84], axis=0)
    q_inj = np.percentile(preds_inj_rec, [16, 50, 84], axis=0)

    #print(np.percentile(data.posterior.data_vars['K'], 84) - np.percentile(data.posterior.data_vars['K'], 50))
    #print(np.percentile(data.posterior.data_vars['K'], 50) - np.percentile(data.posterior.data_vars['K'], 16))
    #print(np.percentile(data.posterior.data_vars['K'], 50))

    # combined residuals 
    res_injected = injected.y - q_inj[1]
    res_training = not_injected.y - q[1]
    res_trainings.append(np.std(res_training))
    res_injecteds.append(np.std(res_injected))

    #plt.scatter(injected.x, injected.y)
    #plt.scatter(not_injected.x, not_injected.y)
    #plt.show()

    #plt.scatter(injected.x, q_inj[1])
    #plt.scatter(not_injected.x, q[1])
    #plt.show()

    #plt.scatter(injected.x, q_inj[1] - injected.y)
    #plt.scatter(not_injected.x, q[1] - not_injected.y)
    #plt.show()

    #print(res_injected)


print("sigma_K from gaspery: ", np.mean(sigma_ks), np.std(sigma_ks))#, np.percentile(sigma_ks, 16), np.percentile(sigma_ks, 50), np.percentile(sigma_ks, 84))
print("sigma_K from MCMC: ", np.mean(sigma_ks_mcmc_median), np.mean(sigma_ks_mcmc_plus), np.mean(sigma_ks_mcmc_minus))
print("sigmas for the sigma_Ks from MCMC: ", np.std(sigma_ks_mcmc_median), np.std(sigma_ks_mcmc_plus), np.std(sigma_ks_mcmc_minus))
print("")
print("training residual: ", np.mean(res_trainings))
print("validation residuals: ", np.mean(res_injecteds))

#plt.errorbar(np.array(injected.x), res_injected, yerr, fmt=".k", capsize=0, label='injected combined', color='r')
#plt.errorbar(np.array(not_injected.x), res_training, yerr, fmt=".k", capsize=0, label='training combined', color='k')
#plt.xlabel("BJD - 2458650 [day]")
#plt.ylim([-25, 25])
#ax4.legend(bbox_to_anchor=(1., 1.05))
#print("std residuals: ", np.std(res_training))
#print("std residuals, injected: ", np.std(res_injected))

#plt.tight_layout()
#plt.savefig(path+'plots/mcmc-strat2-perturbed.png', facecolor='white', bbox_inches='tight')
#plt.show()