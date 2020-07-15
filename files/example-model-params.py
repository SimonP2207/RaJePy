import os
import numpy as np
import scipy.constants as con

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
                   "mlr": 1e-5,
                   },
    "ejection": {"t_0": np.array([0.5]),  # Peak times of bursts (yr)
                 "hl": np.array([0.2]),  # Half-lives of bursts (yr)
                 "chi": np.array([5.0]),  # Burst factors of ejections
                 }
             }

# DO NOT CHANGE BELOW
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
