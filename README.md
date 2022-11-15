### Using the Fisher Information Matrix (FIM) to evaluate different RV observing strategies

#### Case A: white noise, circular orbits
##### calculate_fim.ipynb: initial development of the FIM calculation machinery

##### strategies.ipynb: enable convenient input of n_obs and cadence, with n_obs vs sigma_K as output

##### regimes.ipynb: explore the new 'gappy' regime, in which we introduce dropout and off periods; also codify JAX implementation of FI calculation

##### explore-gappy.ipynb: re-imagine new ways of doing 'gappy' (ie. balanced distribution of off nights) and sensitivity map (on nights vs baseline)

##### au-mic.ipynb: real life example on AU Mic b (and c?)

##### eprv-abstract.ipynb: make cadences vs orbital period sensitivity map, but using latest JAX-enabled FI calculation