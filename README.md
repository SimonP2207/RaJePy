# RaJePy
## Overview
**Ra**dio **Je**ts in **Py**thon (RaJePy) is a Python package which conducts radiative transfer calculations towards a power-law-based, physical model of an ionised jet. Data products from those calculations are subsequently used as the models to conduct synthetic, interferometric radio imaging from.

## Purpose
- Inform radio astronomers on the significance of the detrimental effects of the interferometric imaging of ionised jets
- Allow observers to determine the best telescope configurations and observing frequencies for their science target
- Determine the spectral and morphological evolution of jets with variable ejection events/mass loss rates

### Requirements:
#### Python standard library packages:
- collections.abc
- errno
- os
- pickle (developed with 4.0)
- shutil
- sys
- time
- warnings
#### Other Python packages:
- astropy (developed with 4.0)
- imageio (developed with 2.8.0)
- matplotlib (developed with 3.2.1)
- [mpmath](http://mpmath.org/) (developed with 1.1.0)
- numpy (developed with 1.18.1)
- pandas (developed with 1.0.5)
- scipy (developed with 1.3.1)
#### System-based installations
- Working [casa](https://casa.nrao.edu/) installation (developed with 5.6.2-2) with casa's executable located in `$PATH`

### Future work and direction
- Incorporate inclination into jet model
- Incorporate position angle into jet model
- Parallelise code, especially different synthetic observations and model calculations
