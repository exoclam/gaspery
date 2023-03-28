### Gaspery (/gas.pər.ee/)

[Pronunciation Guide](https://user-images.githubusercontent.com/16911363/212941685-d887b375-176f-4c23-b011-5f6968028a33.mp4)

Gaspery is a package that uses the Fisher Information Matrix (FIM) to evaluate different radial velocity (RV) observing strategies. A paper (Lam, Bedell & Zhao, in prep) will also follow this work.

The Fisher Information Matrix describes the amount of information a time series contains on each free parameter input in a model. For exoplanet observers, the free parameter of interest is often the RV semi-amplitude, K. Gaspery is intended to help observational exoplanet astronomers construct the observing strategy that maximizes information (or minimizes uncertainty) on K. However, one can maximize information on any free parameter from any model, given a time series support (x-axis). So, whether you want to extend gaspery to deal with more complex models and problems or want integration with different covariance kernels (we are fully integrated with tinygp), we are happy to field your suggestions on this repo's Git Issues!

### Installation
To install locally, run 
```
pip install .
``` 
at the root gaspery/ directory. 

To install the latest PyPI distribution of gaspery, on Mac OS or Linux:
```
python3 -m venv some-env
source some-env/bin/activate
python3 -m pip install --index-url https://pypi.org/simple/ --no-deps gaspery
```
And on Windows:
```
py -m venv env
.\env\Scripts\activate
python3 -m pip install --index-url https://pypi.org/simple/ --no-deps gaspery
```

### Tutorials
Start here for usecases! 

- min_observations.ipynb: Given a target, I want to find the minimum number of observations required to reach some threshold for the uncertainty on the RV semi-amplitude, K.

- fixed_budget.ipynb: Given a target AND a fixed allocation of observations, how do I best spend these observing nights to minimize the uncertainty on K? 

- beat_frequencies.ipynb: Given a host star, how do various combinations of planet orbital period and observing cadence interact to produce "beat frequencies" (contours of bad information content due to aliasing)?

- companions.ipynb: Fixed budget, but with more than one planet in a system!

- custom_kernels.ipynb: What if I want to swap out the quasi-periodic Gaussian Process kernel for something more specific to my science case? 


### Src/gaspery
Source code for the gaspery package.


### Notebooks
These are where the functions in src/gaspery/ were developed and tested, as well as where figures were made for proposals, presentations, and the paper.

Notebooks are listed in order of chronological development. This is loosely what the paper follows, as well.

- calculate_fim.ipynb: initial development of the FIM calculation machinery (white noise)

- example.ipynb and example-correlated-noise.ipynb are deep dives into what a strategy looks like over a model planetary signal

- strategies.ipynb: enable convenient input of n_obs and cadence, with n_obs vs sigma_K as output

- regimes.ipynb: explore the new 'gappy' regime, in which we introduce dropout and off periods; also codify JAX implementation of FI calculation

- explore-gappy.ipynb: re-imagine new ways of doing 'gappy' (ie. balanced distribution of off nights) and sensitivity map (on nights vs baseline)

- au-mic.ipynb: real life example for the AU Mic system

- eprv-abstract.ipynb: make cadences vs orbital period sensitivity map, but using latest JAX-enabled FI calculation


### A note on the name
Gaspery is a character from Emily St. John Mandel's Sea of Tranquility who travels throughout time, sampling different points in order to investigate the universe. 
