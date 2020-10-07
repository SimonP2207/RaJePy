"""
Example parameter file for the production of a physical jet model describing the
temperature, density, 3-D velocity and ionisation fraction within a 3-D grid.

Use would be with a RaJePy.classes.JetModel class instance e.g.:

jet_model = RaJePy.classes.JetModel('/full/path/to/example-model-params.py')
"""
import numpy as np
import scipy.constants as con

params = {
    "target": {"name": "M-FH",  # Jet/YSO/... name
               "ra": "06:00:00.00",  # HH:MM:SS.SS... [J2000]
               "dec": "+18:00:00.0",  # DD:MM:SS.SS... [J2000]
               "epoch": "J2000",
               "dist": 700.,  # pc
               "v_lsr": 0.0,  # km/s
               "m_star": 0.5,  # M_sol
               "r_1": .5,  # inner disc radii sourcing the jet in au
               "r_2": 2.5,  # outer disc radii sourcing the jet in au
               },
    "grid": {"n_x": 50,  # No. of cells in x
             "n_y": 400,  # No. of cells in y
             "n_z": 50,  # No. of cells in z
             "l_z": 0.3,  # Length of z-axis in arcsec. Overrides n_x/n_y/n_z.
             "c_size": 1.0,  # Cell size (au)
             },
    "geometry": {"epsilon": 9. / 9.,  # Jet width index
                 "opang": 30.,  # Jet opening angle (deg)
                 "w_0": 5.0,  # Half-width of jet base (au)
                 "r_0": .5,  # Launching radius (au)
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
                   "n_0": None,  # Initial density (cm^-3)
                   "T_0": 1E4,  # Temperature (K)
                   "mu": 1.3,  # Mean atomic weight (m_H)
                   "mlr": 1e-5,  # Msol / yr
                   },
    "ejection": {"t_0": np.array([1200.]),  # Peak times of bursts (yr)
                 "hl": np.array([0.5]),  # Half-lives of bursts (yr)
                 "chi": np.array([2.]),  # Burst factors
                 }
             }

# DO NOT CHANGE BELOW
if params["grid"]["l_z"] is not None:
    i = np.radians(params["geometry"]["inc"])
    pa = np.radians(params["geometry"]["pa"])
    rot_x = np.array([[1., 0., 0.],
                      [0., np.cos(-i), -np.sin(-i)],
                      [0., np.sin(-i), np.cos(-i)]])
    rot_y = np.array([[np.cos(pa), 0., np.sin(pa)],
                      [0., 1., 0.],
                      [-np.sin(pa), 0., np.cos(pa)]])


params["geometry"]["mod_r_0"] = params['geometry']['epsilon'] * \
                                params['geometry']['w_0'] / \
                                np.tan(np.radians(params['geometry']['opang']
                                                  / 2.))
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
    mu = params['properties']['mu'] * 1.673532838e-27  # kg
    w_0 = params['geometry']['w_0'] * con.au  # m
    v_0 = params['properties']['v_0'] * 1000.  # m/s
    n_0 = mlr / (np.pi * w_0**2. * mu * v_0)  # m^-3
    params['properties']['n_0'] = n_0 * 1e-6  # cm^-3
