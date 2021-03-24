# RaJePy
## Overview
**Ra**dio **Je**ts in **Py**thon (RaJePy) is a Python package which conducts radiative transfer calculations towards a power-law-based, physical model of an ionised jet. Data products from those calculations are subsequently used as the models to conduct synthetic, interferometric radio imaging from.

## Purpose
- Inform radio astronomers on the significance of the detrimental effects of the interferometric imaging of ionised jets
- Allow observers to determine the best telescope configurations and observing frequencies for their science target
- Determine the spectral and morphological evolution of jets with variable ejection events/mass loss rates

## Instructions for use
RaJePy's operation is based upon the interplay between the `JetModel` and `Pipeline` classes defined in `classes.py` and initialised in the namespace of the RaJePy module. Basic use can be achieved via the command line with a command of the following syntax:

```python /home/user/RaJePy/main.py -rt -so -v example-model-params.py example-pipeline-params.py```

Help for using the command line can be found by using the command `python /home/user/RaJePy/main.py --help`, which gives the following output:

```
usage: main.py [-h] [-v] [-rt] [-so] [-r] [-c] model_param_file pipeline_param_file

positional arguments:
  model_param_file      Full path to model parameter file
  pipeline_param_file   Full path to pipeline parameter file

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         Increase output verbosity
  -rt, --radiative-transfer
                        Compute radiative transfer solutions
  -so, --simobserve     Conduct synthetic observations using CASA
  -r, --resume          Resume previous pipeline run if present
  -c, --clobber         Overwrite any data products/files present
```

### `JetModel` class
The purpose of the `JetModel` class is to initialise and calculate the physical model grid for an ionised jet.

JetModel's `__init__` method takes one compulsory argument, `params`, and two optional keyword-arguments, `verbose` and `log`:
- `params` : Full path to the model parameters file, an example for which is `RaJePy/files/example-model-params.py` (see a full description below)
- `verbose`: Boolean determining if on an execution of JetModel's class/instance methods, verbose output to the terminal is wanted. Default is `True`.
- `log`: Full path to an optional log file containing a detailed log of all of an instance's method outputs. `None` by default.

For the initialisation of the jet model's physical parameters and their values in each grid cell, the analytical, power law-based approach of [Reynolds (1986)](https://ui.adsabs.harvard.edu/abs/1986ApJ...304..713R/abstract) is used. Those parameters described in that work form the basis of the input parameters required to define a jet model in `RaJePy`.

#### Model parameter file
For the model parameter file, a full-working example is given in `RaJePy/files/example-model-params.py` which contains the following:

```python
import numpy as np
import scipy.constants as con
from scipy.integrate import quad

params = {
    "target": {"name": "LM-N-XW",  # Jet/YSO/Model name
               "ra": "04:27:04.693",  # HH:MM:SS.SS... [J2000]
               "dec": "+26:06:15.82",  # DD:MM:SS.SS... [J2000]
               "epoch": "J2000",
               "dist": 120.,  # pc
               "v_lsr": 0.0,  # km/s
               "M_star": 0.5,  # M_sol in arcsec
               "R_1": .04,  # inner disc radii sourcing the jet in au
               "R_2": .05,  # outer disc radii sourcing the jet in au
               },
    "grid": {"n_x": 50,  # No. of cells in x
             "n_y": 400,  # No. of cells in y
             "n_z": 50,  # No. of cells in z
             "l_z": 4.,  # Length of z-axis. Overrides n_x/n_y/n_z.
             "c_size": 0.25,  # Cell size (au)
             },
    "geometry": {"epsilon": 9. / 9.,  # Jet width index
                 "opang": 20.,  # Jet opening angle (deg)
                 "w_0": 0.25,  # Half-width of jet base (au)
                 "r_0": 0.25,  # Launching radius (au)
                 "inc": 90.,  # Inclination angle (deg)
                 "pa": 0.,  # Jet position PA (deg)
                 },
    "power_laws": {"q_v": 0.,  # Velocity index
                   "q_T": 0.,  # Temperature index
                   "q_x": 0.,  # Ionisation fraction index
                   "q^d_n": -16. / 8., # Cross-sectional density index
                   "q^d_T": 0., # Cross-sectional temperature index
                   "q^d_v": -0.5, # Cross-sectional velocity index
                   "q^d_x": 0.  # Cross-sectional ionisation fraction index
                   },
    "properties": {"v_0": 250.,  # Ejection velocity (km/s)
                   "x_0": 0.1,  # Initial HII fraction
                   "n_0": None,  # Initial density (cm^-3)
                   "T_0": 1E4,  # Temperature (K)
                   "mu": 1.3,  # Mean atomic weight (m_H)
                   "mlr": 1e-8,  # Msol / yr
                   },
    "ejection": {"t_0": np.array([1., 20.]),  # Peak times of bursts (yr)
                 "hl": np.array([0.5, 5.]),  # Half-lives of bursts (yr)
                 "chi": np.array([2., 10.]),  # Burst factors
                 }
             }
# ############################################################################ #
# ####################### DO NOT CHANGE BELOW ################################ #
# ############################################################################ #
# 'Modified' Reynolds ejection radius
params["geometry"]["mod_r_0"] = params['geometry']['epsilon'] * \
                                params['geometry']['w_0'] / \
                                np.tan(np.radians(params['geometry']['opang']
                                                  / 2.))

# Derive power-law indices for number density and optical depths as functions
# of distance along the jet axis, r
params["power_laws"]["q_n"] = -params["power_laws"]["q_v"] - \
                              (2.0 * params["geometry"]["epsilon"])
params["power_laws"]["q_tau"] = params["geometry"]["epsilon"] + \
                                2.0 * params["power_laws"]["q_x"] + \
                                2.0 * params["power_laws"]["q_n"] - \
                                1.35 * params["power_laws"]["q_T"]

# Derive initial number density of jet given the defined mass-loss rate by:
# 1.) Integrating r_eff from 0 --> w_0 to give 'effective area' of jet base
def n_w(w_0, R_1, R_2, q_nd, q_vd):
    """Decorator function of integrand for number density, n, as a function of
    jet width, w"""
    def func(w):
        return 2. * np.pi * w * (1 + (w * (R_2 - R_1)) /
                                 (R_1 * w_0)) ** (q_nd + q_vd)
    return func

f = n_w(params["geometry"]["w_0"] * con.au * 1e2,
        params["target"]["R_1"] * con.au * 1e2,
        params["target"]["R_2"] * con.au * 1e2,
        params["power_laws"]["q^d_n"], params["power_laws"]["q^d_v"])
result = quad(f, 0, params["geometry"]["w_0"] * con.au * 1e2)

# 2.) Use given mass loss rate divided by initial velocity to calculate
# initial number density
mu = 1.673532838e-27 * params["properties"]["mu"] # average particle mass in kg
ndot = params['properties']['mlr'] * 1.989e30 / con.year / mu  # particles / s
v_0 = params['properties']['v_0'] * 1e5  # cm / s
params['properties']['n_0'] = ndot / (result[0] * v_0)  # cm^-3
```
This shows a python `dict` containing 6 keys associated with more nested `dict`s. For each of those 6 keys, their logical purpose and description of their associated values' `dict`'s keys are given in separate tables below.

##### `params['target']`
'Science' target information.

| Parameter/key | Description                                                         | Type    | Example         |
|---------------|---------------------------------------------------------------------|---------|-----------------|
| `"name"`      | Target object name                                                  | `str`   | `"S255 IRS3"`   |
| `"ra"`        | Target object Right Ascension (HH:MM:SS.S)                          | `str`   | `"01:23:45.67"` |
| `"dec"`       | Target object Declination (DD:MM:SS.S)                              | `str`   | `"-87:65:43.2"` |
| `"epoch"`     | Epoch of RA/Dec                                                     | `str`   | `"J2000"`       |
| `"dist"`      | Distance (pc)                                                       | `float` | `1780.`         |
| `"v_lsr"`     | Systemic velocity with respect to the Local Standard of Rest (km/s) | `float` | `-4.2`          |
| `"M_star"`    | Central protostellar mass (solar masses).                           | `float` | `10.0`          |
| `"r_1"`       | Inner disc radius from which jet material is sources (au)           | `float` | `1.0`           |
| `"r_2"`       | Outer disc radius from which jet material is sources (au)           | `float` | `10.0`          |


##### `params['grid']`
Model grid dimensions

| Parameter/key | Description                                                                               | Type          | Example |
|---------------|-------------------------------------------------------------------------------------------|---------------|---------|
| `"n_x"`       | Number of cells in x-axis                                                               | `int`         | 100     |
| `"n_y"`       | Number of cells in y-axis                                                               | `int`         | 100     |
| `"n_z"`       | Number of cells in z-axis                                                               | `int`         | 400     |
| `"l_z"`       | Full length of z-axis/bi-polar jet (arcsec). Overrides `"n_x"`/`"n_y"`/`"n_z"` parameters | `float`, None | 2.0     |
| `"c_size"`    | Grid cell size (au)                                                                       | `float`       | 2.0     |

**NB** - If not `None`, `"l_z"` calculates (using supplied `"dist"` and `"c_size"`) and updates `"n_x"`/`"n_y"`/`"n_z"` parameters to fully encompass a jet of length `"l_z"` arcseconds.

##### `params['geometry']`
Jet geometry parameters

| Parameter/key | Description                                                        | Type    | Example  |
|---------------|--------------------------------------------------------------------|---------|----------|
| `"epsilon"`   | Power-law coefficient for jet width                                | `float` | `+1.0`   |
| `"opang"`     | Opening angle (deg)                                                | `float` | `20.0`   |
| `"w_0"`       | Jet half-width at jet base (au)                                    | `float` | `2.0`    |
| `"r_0"`       | Launching radius (au)                                              | `float` | `4.0`    |
| `"inc"`       | Jet inclination (deg)                                              | `float` | `90.`    |
| `"pa"`        | Jet position angle (deg)                                           | `float` | `0.`     |

##### `params['power_laws']`

| Parameter/key | Description                                                                               | Type    | Example |
|---------------|-------------------------------------------------------------------------------------------|---------|---------|
| `"q_v"`       | Power-law coefficient for jet velocity as function of r                                   | `float` | `-0.5`  |
| `"q_T"`       | Power-law coefficient for jet temperature as function of r                                | `float` | `-0.5`  |
| `"q_x"`       | Power-law coefficient for jet ionisation fraction as function of r                        | `float` | `0.0`   |
| `"q^d_n"`     | Cross-sectional power-law coefficient for jet number density as function of w             | `float` | `-2.0`  |
| `"q^d_T"`     | Cross-sectional power-law coefficient for jet electron temperature as function of w       | `float` | `-2.0`  |
| `"q^d_v"`     | Cross-sectional power-law coefficient for jet velocity as function of w                   | `float` | `-0.5`  |
| `"q^d_x"`     | Cross-sectional power-law coefficient for jet ionisation fraction as function of w        | `float` | `-2.0`  |

##### `params['properties']`
Jet physical parameter values

| Parameter/key | Description                                              | Type          | Example |
|---------------|----------------------------------------------------------|---------------|---------|
| `"v_0"`       | Jet initial velocity (km/s)                              | `float`       | `500.`  |
| `"x_0"`       | Initial jet ionisation fraction (0 --> 1                 | `float`       | `0.1`   |
| `"n_0"`       | Initial jet number density (per cubic cm)                | `float`, None | `1e9`   |
| `"T_0"`       | Initial jet temperature (K)                              | `float`       | `1e4`   |
| `"mu"`        | Mean atomic weight of jet (hydrogen atom mass)           | `float`       | `1.3`   |
| `"mlr"`       | Jet mass loss rate (solar masses per yr)                 | `float`, None | `1e-5`  |

**NB** - `"mlr"` overrides `"n_0"` if it is not `None` and calculates/updates `"n_0"` to give the required mass loss rate.

##### `params['ejection']`
Jet mass loss variability parameters

| Parameter/key | Description                                                        | Type                        | Example                        |
|---------------|--------------------------------------------------------------------|-----------------------------|--------------------------------|
| `"t_0"`       | Burst(s) peak times (yr)                                           | `numpy.array` (dtype=float) | `numpy.array([0., 1., 2.])`    |
| `"hl"`        | Burst(s) 'half-lives', i.e. FWHM in time (yr)                      | `numpy.array` (dtype=float) | `numpy.array([0.2, 0.1, 0.8])` |
| `"chi"`       | Burst(s) factors (multiple of jet's steady state mass loss rate    | `numpy.array` (dtype=float) | `numpy.array([10., 5., 2.])`   |

Other lines of code at the bottom of `example-model-params.py` (below the comment `# DO NOT CHANGE BELOW!`) derive various required jet parameters. **Please do not change those lines!**

### `Pipeline` class
The purpose of the `Pipeline` class is to handle directory/file manipulation and perform synthetic observations via [casa](https://casa.nrao.edu/), their subsequent measurements and other analyses conducted on a `JetModel` instance.

`Pipeline`'s `__init__` method takes two compulsory arguments, `jetmodel` and `params`:
- `jetmodel`: `JetModel` instance to conduct all analyses and observations upon
- `params` : Full path to the pipeline parameters file, an example for which is `RaJePy/files/example-pipeline-params.py` (see a full description below)

#### Pipeline parameter file
For the pipeline parameter file, a full-working example is given in `RaJePy/files/example-pipeline-params.py` which contains the following:

```python
import os
import numpy as np

params = {'min_el':    20.,    # Min. elevation for synthetic observations (deg)
          'dcys':      {"model_dcy": os.sep.join(['mnt', 'purser_data', 'RaJePy',
                                                  "LM-N-XW"])},
          # Continuum observations
          'continuum': {'times':  np.array([0., 0.2, 0.4, 0.6, 0.8, 1., 1.2,
                                            1.4, 1.6, 1.8, 2.0, 3.0, 4.0, 5.0,
                                            6.0, 10., 12., 14., 16., 18., 19.,
                                            20., 21., 22., 24., 26., 28.,
                                            30.]),  # yr
                        'freqs':  np.array([1.5, 3.0, 6., 10., 22., 33.,
                                            43.]) * 1e9,
                        't_obs':  np.array([1200, 1200, 3600, 1200,
                                            1200, 1800, 2400]),
                        'tscps':  np.array([('VLA', 'A'), ('VLA', 'A'),
                                            ('VLA', 'A'), ('VLA', 'A'),
                                            ('VLA', 'A'), ('VLA', 'A'),
                                            ('VLA', 'A')]),
                        't_ints': np.array([5, 5, 5, 5, 5, 5, 5]),    # secs
                        'bws':    np.array([1e9, 2e9, 2e9, 4e9, 4e9,
                                            4e9, 8e9]),  # Hz
                        'chanws': np.array([1e8] * 7)},  # int
          # Radio recombination line observations
          'rrls':      {'times':  np.array([0., 0.2, 0.4, 0.6, 0.8, 1., 1.2,
                                            1.4, 1.6, 1.8, 2.0, 3.0, 4.0, 5.0,
                                            6.0, 10., 12., 14., 16., 18., 19.,
                                            20., 21., 22., 24., 26., 28.,
                                            30.]),  # yr
                        'lines':  np.array(["H75a", "H65a", "H58a", "H52a"]),  # str (Element+n+dn)
                        't_obs':  np.array([36000, 36000, 36000, 36000]),  # secs
                        'tscps':  np.array([('VLA', 'A'), ('VLA', 'A'), ('VLA', 'A'), ('VLA', 'A')]),  # (tscop, config)
                        't_ints': np.array([60, 60, 60, 60]),  # secs
                        'bws':    np.array([128e6, 128e6, 128e6, 128e6]),  # Hz
                        'chanws': np.array([1e6, 1e6, 1e6, 1e6])},  # Hz
          }
```
This shows a python `dict` containing 4 keys associated keys which are:

| Parameter/key | Description                                                    | Type    | Example   |
|---------------|----------------------------------------------------------------|---------|-----------|
| `"min_el"`    | Minimum elevation to conduct synthetic observations (deg)      | `float` | `20.`     |
| `"dcys  "`    | Contains relevant directories needed                           | `float` | `20.`     |
| `"continuum"` | Contains all continuum run specifications                      | `dict`  | See below |
| `"rrls"`      | Contains all radio recombination line run specifications       | `dict`  | See below |

Within the dict `params['dcys']`, the following directories are specified:

| Parameter/key | Description                                                                     | Type  | Example                     |
|---------------|---------------------------------------------------------------------------------|-------|-----------------------------|
| `'model_dcy'` | Full path for all products and directories associated with `Pipeline` execution | `str` | `/my/rajepy/exec/directory` |

Both the `params['continuum']` and `params['rrls']` dicts share the following keys/values whose logical purpose and description of their associated values are given in the table below:

| Parameter/key | Description                                                    | Type                                       | Example                                                                     |
|---------------|----------------------------------------------------------------|--------------------------------------------|-----------------------------------------------------------------------------|
| `"times"`     | Observational times (yr)                                       | `numpy.array` with `dtype=float`           | `numpy.linspace(0., 5., 4)`                                                 |
| `"t_obs"`     | Total scan times (s)                                           | `numpy.array` with `dtype=int`             |                                                                             |
| `"tscps"`     | Telescope names and configurations                             | `numpy.array` with `dtype=tuple(str, str)` | `numpy.array([('VLA', 'A'), ('EMERLIN', '0'), ('VLA', 'B'), ('VLA', 'C')])` |
| `"t_ints"`    | Visibility integration times (s)                               | `numpy.array` with `dtype=int`             | `numpy.array([5, 3, 3, 2])`                                                 |
| `"bws"`       | Bandwidths (Hz)                                                | `numpy.array` with `dtype=float`           | `numpy.array([0.5e9, 2e9, 2e9, 4e9])`                                       |
| `"chanws"`    | Channel widths within bandwidth (Hz)                           | `numpy.array` with `dtype=int`             | `numpy.array([1, 1, 1, 1])`                                                 |
| `"min_el"`    | Minimum elevation to conduct synthetic observations (deg)      | `float`                                    | `20.`                                                                       |

However, each type of observation differs in how the observing frequencies are specified. For `params['continuum']`, the central observing frequency is directly specified via:

| Parameter/key | Description                                                    | Type                                       | Example                                                                     |
|---------------|----------------------------------------------------------------|--------------------------------------------|-----------------------------------------------------------------------------|
| `"freqs"`     | Observational frequencies (Hz)                                 | `numpy.array` with `dtype=float`           | `numpy.array([1e9, 5e9, 2e10, 5e10])`                                       |

While for `params['rrls']`, the relevant radio recombination line is specified (e.g. `H57a`), and RaJePy automatically calculates the central observing frequency based upon that:

| Parameter/key | Description                             | Type                                       | Example                                                                     |
|---------------|-----------------------------------------|--------------------------------------------|-----------------------------------------------------------------------------|
| `"lines"`     | Radio recombination lines for observing | `numpy.array` with `dtype=str`           | `numpy.array(['H56a', 'H42a', 'H76a'])`                                       |

### Executing a complete pipeline
After creating instances of the `JetModel` and `Pipeline` classes, execution of the desired synthetic observations etc. takes place via the `Pipeline` class' `execute` method. A complete script for execution of a synthetic observing and model calculation run would be:

```python
import os
import RaJePy as rjp

model_params = 'model-params.py'
pline_params = 'pipeline-params.py'
log_file = 'TestJet.log'

jm = rjp.JetModel(model_params, log=log_file)
pline = rjp.Pipeline(jm, pline_params)

pline.execute(simobserve=True, dryrun=False)
```

#### `Pipeline.execute` method
The `execute` method runs the complete pipeline, producing relevant plots, `.fits` model files, measurement sets and final clean image `.fits` files. It takes 5, optional, keyword-arguments:

| `kwarg`      | Description                                                                                  | Type   |
|--------------|----------------------------------------------------------------------------------------------|--------|
| `simobserve` | Whether to conduct synthetic observations on the model `.fits` file                          | `bool` |
| `verbose`    | Verbose output to the terminal?                                                              | `bool` |
| `dryrun`     | Whether to execute a dry run, without actually running any calculations (for testing)        | `bool` |
| `resume`     | Whether to resume a previously saved run (if saved model file and saved pipeline file exist) | `bool` |
| `clobber`    | Whether to redo and overwrite previously written files [soon to be deprecated]               | `bool` |

### Data products
If `/example/dcy` was given for `params['dcys']['model_dcy']` in the pipeline parameter file, the output from the above code would be:

```sh
/example/dcy/  # Set by user in pipeline parameter file's params['dcys']['model_dcy']
├──ModelRun_YYYY-MM-DD-HH:MM:SS.log  # Generated log file by Pipeline
├──jetmodel.save  # Saved JetModel instance (large file size!)
├──modelrun.save  # Saved Pipeline instance
├──pointings.ptg  # Synthetic observation pointing file
├──Day0/  # First epoch directory for modelling/observations
│  ├──ModelPlot.pdf  # Plot of resulting physical model
│  ├──1000MHz/  # Directory containing data products for first radio frequency
│  │  ├──RadioPlot.pdf  # Plot of model fluxes, optical depths and emission measures
│  │  ├──DDMMYYYY_HHMMSS.py  # Casa script to be executed, generated by Pipeline
│  │  ├──DDMMYYYY_HHMMSS.log  # Generated logfile by JetModel
│  │  ├──EM_Day0_1000MHz.fits  # Model emission measure image
│  │  ├──Flux_Day0_1000MHz.fits  # Model radio flux image
│  │  ├──Tau_Day0_1000MHz.fits  # Model optical depth image
│  │  ├──SynObs.vla.a.noisy.imaging.fits  # Final clean image
│  │  ├──SynObs.vla.a.noisy.imaging.estimates  # Input to casa.imfit
│  │  ├──SynObs.vla.a.noisy.imaging.imfit  # Results of Gaussian fits to jet
│  │  └──SynObs/  # All files generated by CASA
│  │     ├──SynObs.vla.a.ms/  # No-noise measurement set (casa.simobserve)
│  │     ├──SynObs.vla.a.noisy.ms/  # Noisy measurement set (casa.simobserve)
│  │     ├──SynObs.vla.a.noisy.imaging.image/  # Clean image (casa.tclean)
│  │     ├──SynObs.vla.a.noisy.imaging.mask/  # Clean mask (casa.tclean)
│  │     ├──SynObs.vla.a.noisy.imaging.model/  # Deconvolved model image (casa.tclean)
│  │     ├──SynObs.vla.a.noisy.imaging.pb/  # Primary beam image (casa.tclean)
│  │     ├──SynObs.vla.a.noisy.imaging.psf/  # Point spread function image (casa.tclean)
│  │     ├──SynObs.vla.a.noisy.imaging.residual/  # Residual map (casa.tclean)
│  │     ├──SynObs.vla.a.noisy.imaging.sumwt/  # Sum-of-weights image (casa.tclean)
│  │     ├──SynObs.vla.a.skymodel/  # Input model image (casa.simobserve)
│  │     └──SynObs.vla.a.skymodel.flat/  # Flattened input model image (casa.simobserve)
│  ├──5000MHz/  # Directory containing data products for second radio frequency
│  |  └──...
│  ├──20000MHz/  # Directory containing data products for third radio frequency
│  |  └──...
│  └──50000MHz/  # Directory containing data products for fourth radio frequency
│     └──...
├──Day100/  # Second epoch directory for modelling/observations
│  └──...
└──...
```

## Requirements:
### Python standard library packages:
- collections.abc
- errno
- os
- pickle (developed with 4.0)
- shutil
- sys
- time
- warnings
### Other Python packages:
- astropy (developed with 4.0)
- imageio (developed with 2.8.0)
- matplotlib (developed with 3.2.1)
- [mpmath](http://mpmath.org/) (developed with 1.1.0)
- numpy (developed with 1.18.1)
- pandas (developed with 1.0.5)
- scipy (developed with 1.3.1)
- tabulate (developed with 0.8.9)
### System-based installations
- Working [casa](https://casa.nrao.edu/) installation (developed with 6.1.2.7) with casa's executable located in `$PATH`

## Future work and direction
- ~~Incorporate inclination into jet model~~
- ~~Incorporate position angle into jet model~~
- ~~Implement more than one channel across bandwidth for more accurate multi-frequency synthesis~~
- Parallelise code, especially different synthetic observations and model calculations
