### Using the Fisher Information Matrix (FIM) to evaluate different RV observing strategies

#### Case A: white noise, circular orbits
##### calculate_fim.ipynb: initial development of the FIM calculation machinery
- Need to go from element-by-element calculation to array-wide calculation
- For the foreseeable future, plot both inv(FIM)\_ii and 1/np.sqrt(FIM\_ii)

##### strategies.ipynb: enable convenient input of n_obs and cadence, with n_obs vs sigma_K as output
