# RaJePy
## Overview
**Ra**dio **Je**ts in **Py**thon (RaJePy) is a Python package which conducts radiative transfer calculations towards a power-law-based, physical model of an ionised jet. Data products from those calculations are subsequently used as the models to conduct synthetic, interferometric radio imaging from.

## Purpose
- Inform radio astronomers on the significance of the detrimental effects of the interferometric imaging of ionised jets
- Allow observers to determine the best telescope configurations and observing frequencies for their science target
- Determine the spectral and morphological evolution of jets with variable ejection events/mass loss rates

## Instructions for use
RaJePy's operation is based upon the interplay between the `JetModel` and `ModelRun` classes defined in `classes.py` and initialised in the namespace of the RaJePy module.

### `JetModel` class
The purpose of the `JetModel` class is to intialise and calculate the physical model grid for an ionised jet.

JetModel's `__init__` method takes one compulsory argument, `params`, and two optional keyword-arguments, `verbose` and `log`:
- `params` : Full path to the model parameters file, an example for which is `RaJePy/files/example-model-params.py` (see a full description below)
- `verbose`: Boolean determining if on an execution of JetModel's class/instance methods, verbose output to the terminal is wanted. Default is `True`.
- `log`: Full path to an optional log file containing a detailed log of all of an instance's method outputs. `None` by default.

For the initialisation of the jet model's physical parameters and their values in each grid cell, the analytical, power law-based approach of [Reynolds (1986)](https://ui.adsabs.harvard.edu/abs/1986ApJ...304..713R/abstract) is used. Those parameters described in that work form the basis of the input parameters required to define a jet model in `RaJePy`.

#### Model parameter file
For the model parameter file, a full-working example is given in `RaJePy/files/example-model-params.py` which contains the following:

```python
params = {
    "target": {"name": "S255IRS3",  # Jet/YSO/... name
               "ra": "06:12:54.02",  # HH:MM:SS.SS... [J2000]
               "dec": "+17:59:23.6",  # DD:MM:SS.SS... [J2000]
               "epoch": "J2000",
               "dist": 1780.,  # pc
               "v_lsr": 7.4,  # km/s
               "m_star": 1.0,  # M_sol
               "r_1": 0.1,  # inner disc radii sourcing the jet in au
               "r_2": 1.0,  # outer disc radii sourcing the jet in au
               },
    "grid": {"n_x": 40,  # No. of cells in x
             "n_y": 40,  # No. of cells in y
             "n_z": 100,  # No. of cells in z
             "l_z": 2.0,  # Length of z-axis in arcsec. Overrides n_x/n_y/n_z.
             "c_size": 2.0,  # Cell size (au)
             },
    "geometry": {"epsilon": 9. / 9.,  # Jet width index
                 "w_0": 2.0,  # Half-width of jet base (au)
                 "r_0": 4.0,  # Launching radius (au)
                 "inc": 90.,  # Inclination angle (deg)
                 "pa": 0.,  # Jet position PA (deg)
                 "exp_cs": False,  # Transverse exp. density profile?
                 },
    "power_laws": {"q_v": 0.,  # Velocity index
                   "q_T": 0.,  # Temperature index
                   "q_x": 0.,  # HII fraction index
                   },
    "properties": {"v_0": 500.,  # Ejection velocity (km/s)
                   "x_0": 0.1,  # Initial HII fraction
                   "n_0": 2.6e9,  # Initial density (cm^-3)
                   "T_0": 1E4,  # Temperature (K)
                   "mu": 1.3,  # Mean atomic weight (m_H)
                   "mlr": 1e-5,  # Mass loss rate (M_sol / yr)
                   },
    "ejection": {"t_0": np.array([0.5]),  # Peak times of bursts (yr)
                 "hl": np.array([0.2]),  # Half-lives of bursts (yr)
                 "chi": np.array([5.0]),  # Burst factors of ejections
                 }
             }

# DO NOT CHANGE BELOW!
params["power_laws"]["q_n"] = -params["power_laws"]["q_v"] - \
                              (2.0 * params["geometry"]["epsilon"])
params["power_laws"]["q_tau"] = params["geometry"]["epsilon"] + \
                                2.0 * params["power_laws"]["q_x"] + \
                                2.0 * params["power_laws"]["q_n"] - \
                                1.35 * params["power_laws"]["q_T"]
if params['properties']['mlr'] is None:
    mlr = params['properties']['n_0'] * 1e6 * np.pi
    mlr *= params['properties']['mu'] * 1.67353e-27
    mlr *= (params['geometry']['w_0'] * con.au)**2.
    mlr *= params['properties']['v_0'] * 1e3  # kg/s
    mlr *= con.year / 1.98847e30  # Msol/yr
    params['properties']['mlr'] = mlr
else:
    mlr = params['properties']['mlr'] * 1.98847e30 / con.year  # kg/s
    mu = params['properties']['mu'] * 1.67353e-27  # kg
    w_0 = params['geometry']['w_0'] * con.au  # m
    v_0 = params['properties']['v_0'] * 1000.  # m/s
    n_0 = mlr / (np.pi * w_0**2. * mu * v_0)  # m^-3
    params['properties']['n_0'] = n_0 * 1e-6  # cm^-3
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
| `"m_star"`    | Central protostellar mass (![equation](http://www.sciweavers.org/tex2img.php?eq=%7B%5Crm%20M%7D_%5Codot&bc=Transparent&fc=Black&im=png&fs=12&ff=modern&edit=0))                         | `float` | `10.0`          |
| `"r_1"`       | Inner disc radius from which jet material is sources (au)           | `float` | `1.0`           |
| `"r_2"`       | Outer disc radius from which jet material is sources (au)           | `float` | `10.0`          |


##### `params['grid']`
Model grid dimensions

| Parameter/key | Description                                                                               | Type          | Example |
|---------------|-------------------------------------------------------------------------------------------|---------------|---------|
| `"n_x"`       | Number of cells in $$x$$-axis                                                               | `int`         | 100     |
| `"n_y"`       | Number of cells in $y$-axis                                                               | `int`         | 100     |
| `"n_z"`       | Number of cells in $z$-axis                                                               | `int`         | 400     |
| `"l_z"`       | Full length of z-axis/bi-polar jet (arcsec). Overrides `"n_x"`/`"n_y"`/`"n_z"` parameters | `float`, None | 2.0     |
| `"c_size"`    | Grid cell size (au)                                                                       | `float`       | 2.0     |

**NB** - If not `None`, `"l_z"` calculates (using supplied `"dist"` and `"c_size"`) and updates `"n_x"`/`"n_y"`/`"n_z"` parameters to fully encompass a jet of length `"l_z"` arcseconds.

##### `params['geometry']`
Jet geometry parameters

| Parameter/key | Description                                                        | Type    | Example  |
|---------------|--------------------------------------------------------------------|---------|----------|
| `"epsilon"`   | Power-law coefficient for jet width i.e. ![equation](http://www.sciweavers.org/tex2img.php?eq=w%20%5Cleft%28%20r%20%5Cright%29%20%5Cpropto%20r%5E%7B%5Cepsilon%7D&bc=Transparent&fc=Black&im=png&fs=12&ff=modern&edit=0) | `float` | `+1.0`   |
| `"w_0"`       | Jet half-width at jet base (au)                                    | `float` | `2.0`    |
| `"r_0"`       | Launching radius (au)                                              | `float` | `4.0`    |
| `"inc"`       | Jet inclination (deg) [not implemented]                            | `float` | `90.`    |
| `"pa"`        | Jet position angle (deg) [not implemented]                         | `float` | `0.`     |
| `"exp_cs"`    | Exponential density cross-section? [not implemented]               | `bool`  | `False`  |

##### `params['power_laws']`

| Parameter/key | Description                                                                               | Type    | Example |
|---------------|-------------------------------------------------------------------------------------------|---------|---------|
| `"q_v"`       | Power-law coefficient for jet velocity i.e. ![equation](http://www.sciweavers.org/tex2img.php?eq=v%28r%29%20%5Cpropto%20r%5E%7B%5Crm%20q_v%7D&bc=Transparent&fc=Black&im=png&fs=12&ff=modern&edit=0)                   | `float` | `-0.5`  |
| `"q_T"`       | Power-law coefficient for jet temperature i.e. ![equation](http://www.sciweavers.org/tex2img.php?eq=T%28r%29%20%5Cpropto%20r%5E%7B%5Crm%20q_T%7D&bc=Transparent&fc=Black&im=png&fs=12&ff=modern&edit=0)                | `float` | `-0.5`  |
| `"q_x"`       | Power-law coefficient for jet ionisation fraction i.e. ![equation](http://www.sciweavers.org/tex2img.php?eq=%5Cchi_i%20%28r%29%20%5Cpropto%20r%5E%7B%5Crm%20q_%5Cchi%7D&bc=Transparent&fc=Black&im=png&fs=12&ff=modern&edit=0) | `float` | `0.0`   |

##### `params['properties']`
Jet physical parameter values

| Parameter/key | Description                                              | Type    | Example |
|---------------|----------------------------------------------------------|---------|---------|
| `"v_0"`       | Jet initial velocity (km/s)                 | `float` | `500.`  |
| `"x_0"`       | Initial jet ionisation fraction (0 --> 1) | `float` | `0.1`   |
| `"n_0"`       | Initial jet number density (![equation](http://www.sciweavers.org/tex2img.php?eq=%7B%5Crm%20cm%5E%7B-3%7D%7D&bc=Transparent&fc=Black&im=png&fs=12&ff=modern&edit=0))             | `float` | `1e9`   |
| `"T_0"`       | Initial jet temperature (K)                              | `float` | `1e4`   |
| `"mu"`        | Mean atomic weight of jet (![equation](http://www.sciweavers.org/tex2img.php?eq=%7B%5Crm%20m_%7BH%7D%7D&bc=Transparent&fc=Black&im=png&fs=12&ff=modern&edit=0))                | `float` | `1.3`   |
| `"mlr"`       | Jet mass loss rate (![equation](http://www.sciweavers.org/tex2img.php?eq=%7B%5Crm%20M_%5Codot%20%5C%2C%20yr%5E%7B-1%7D%7D&bc=Transparent&fc=Black&im=png&fs=12&ff=modern&edit=0))          | `float`, None | `1e-5`  |

**NB** - `"mlr"` overrides `"n_0"` if it is not `None` and calculates and updates `"n_0"` to give the required mass loss rate.

##### `params['ejection']`
Jet mass loss variability parameters

| Parameter/key | Description                                                        | Type                        | Example                        |
|---------------|--------------------------------------------------------------------|-----------------------------|--------------------------------|
| `"t_0"`       | Burst peak times (yr)                                              | `numpy.array` (dtype=float) | `numpy.array([0., 1., 2.])`    |
| `"hl"`        | Burst `half-lives`, i.e. FWHM in time (yr)                         | `numpy.array` (dtype=float) | `numpy.array([0.2, 0.1, 0.8])` |
| `"chi"`       | Burst factors (multiple of jet's steady state ![equation](http://www.sciweavers.org/tex2img.php?eq=%5Cdot%7BM%7D_%7B%5Crm%20jet%7D&bc=Transparent&fc=Black&im=png&fs=12&ff=modern&edit=0)) | `numpy.array` (dtype=float) | `numpy.array([10., 5., 2.])`   |

Other lines of code at the bottom of `example-model-params.py` (below the comment `# DO NOT CHANGE BELOW!`) derive various required jet parameters. **Please do not change those lines!** As for the required model parameters


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
