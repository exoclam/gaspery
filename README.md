### Gaspery (/gas.pər.ee/) https://user-images.githubusercontent.com/16911363/212937197-14b049b0-9c01-46cc-aec0-fc4c0e1dd0be.mp4

Gaspery is a package that uses the Fisher Information Matrix (FIM) to evaluate different radial velocity (RV) observing strategies. It is currently only locally installable (run "pip install ." at the root gaspery/ directory), but will soon be on PyPI! A paper (Lam, Bedell & Zhao, in prep) will also follow this work.

The Fisher Information Matrix describes the amount of information a time series contains on each free parameter input in a model. For exoplanet observers, the free parameter of interest is probably the RV semi-amplitude (sometimes called 'K'). Gaspery is intended to serve observational astronomers in their search for the observing strategy that maximizes information (or minimizes uncertainty) on K. However, one can maximize information on any free parameter from any model, given a time series support (x-axis). So, whether you want to extend gaspery to deal with more complex models or kernels, we are happy to field your suggestions on this repo's Git Issues!

The directory layout is as follows: 

#### Tutorials
Start here for usecases! 

- min_observations.ipynb: Given a target, I want to find the minimum number of observations required to reach some threshold for the uncertainty on the RV semi-amplitude, K.

- fixed_budget.ipynb: Given a target AND a fixed allocation of observations, how do I best spend these observing nights to minimize the uncertainty on K? 


#### Src/gaspery
Source code for the gaspery package.


#### Notebooks
These are where the functions in src/gaspery/ were developed and tested, as well as where figures were made for proposals, presentations, and the paper.

Notebooks are listed in order of chronological development. This is loosely what the paper follows, as well.

- calculate_fim.ipynb: initial development of the FIM calculation machinery (white noise)

- example.ipynb and example-correlated-noise.ipynb are deep dives into what a strategy looks like over a model planetary signal

- strategies.ipynb: enable convenient input of n_obs and cadence, with n_obs vs sigma_K as output

- regimes.ipynb: explore the new 'gappy' regime, in which we introduce dropout and off periods; also codify JAX implementation of FI calculation

- explore-gappy.ipynb: re-imagine new ways of doing 'gappy' (ie. balanced distribution of off nights) and sensitivity map (on nights vs baseline)

- au-mic.ipynb: real life example for the AU Mic system

- eprv-abstract.ipynb: make cadences vs orbital period sensitivity map, but using latest JAX-enabled FI calculation
