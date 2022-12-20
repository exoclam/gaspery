### Using the Fisher Information Matrix (FIM) to evaluate different RV observing strategies

A paper (Lam, Bedell & Zhao, in prep) will follow this work.

#### Notebooks
These are where the functions in src/gaspery/ were developed and tested, as well as where figures were made for proposals, presentations, and the paper.

Notebooks are listed in order of chronological development. This is loosely what the paper follows, as well.

##### calculate_fim.ipynb: initial development of the FIM calculation machinery (white noise)

##### example.ipynb and example-correlated-noise.ipynb are deep dives into what a strategy looks like over a model planetary signal

##### strategies.ipynb: enable convenient input of n_obs and cadence, with n_obs vs sigma_K as output

##### regimes.ipynb: explore the new 'gappy' regime, in which we introduce dropout and off periods; also codify JAX implementation of FI calculation

##### explore-gappy.ipynb: re-imagine new ways of doing 'gappy' (ie. balanced distribution of off nights) and sensitivity map (on nights vs baseline)

##### au-mic.ipynb: real life example for the AU Mic system

##### eprv-abstract.ipynb: make cadences vs orbital period sensitivity map, but using latest JAX-enabled FI calculation