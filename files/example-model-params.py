"""
Example parameter file for the production of a physical jet model describing the
temperature, density, 3-D velocity and ionisation fraction within a 3-D grid.

Use would be with a RaJePy.classes.JetModel class instance e.g.:

jet_model = RaJePy.classes.JetModel('/full/path/to/example-model-params.py')
"""
import numpy as np
import scipy.constants as con
from scipy.integrate import quad

params = {
    "target": {"name": "M-TM1",  # Jet/YSO/Model name
               "ra": "0:00:00.00",  # HH:MM:SS.SS... [J2000]
               "dec": "+0:00:00.0",  # DD:MM:SS.SS... [J2000]
               "epoch": "J2000",
               "dist": 1000.,  # pc
               "v_lsr": 0.0,  # km/s
               "M_star": 10.,  # M_sol in arcsec
               "R_1": .5,  # inner disc radii sourcing the jet in au
               "R_2": 2.5,  # outer disc radii sourcing the jet in au
               },
    "grid": {"n_x": 50,  # No. of cells in x
             "n_y": 400,  # No. of cells in y
             "n_z": 50,  # No. of cells in z
             "l_z": 1.,  # Length of z-axis. Overrides n_x/n_y/n_z.
             "c_size": 2.5,  # Cell size (au)
             },
    "geometry": {"epsilon": 9. / 9.,  # Jet width index
                 "opang": 30.,  # Jet opening angle (deg)
                 "w_0": 2.5,  # Half-width of jet base (au)
                 "r_0": 1.,  # Launching radius (au)
                 "inc": 90.,  # Inclination angle (deg)
                 "pa": 0.,  # Jet position PA (deg)
                 },
    "power_laws": {"q_v": 0.,  # Velocity index
                   "q_T": 0.,  # Temperature index
                   "q_x": 0.,  # Ionisation fraction index
                   "q^d_n": 0. / 8., # Cross-sectional density index
                   "q^d_T": 0., # Cross-sectional temperature index
                   "q^d_v": 0., # Cross-sectional velocity index
                   "q^d_x": 0.  # Cross-sectional ionisation fraction index
                   },
    "properties": {"v_0": 500.,  # Ejection velocity (km/s)
                   "x_0": 0.1,  # Initial HII fraction
                   "n_0": None,  # Initial density (cm^-3)
                   "T_0": 1E4,  # Temperature (K)
                   "mu": 1.3,  # Mean atomic weight (m_H)
                   "mlr": 1e-5,  # Msol / yr
                   },
    "ejection": {"t_0": np.array([145.]),  # Peak times of bursts (yr)
                 "hl": np.array([0.1]),  # Half-lives of bursts (yr)
                 "chi": np.array([10.]),  # Burst factors
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
# ############################################################################ #
if __name__ == '__main__':
    from RaJePy.miscellaneous import functions
    print(functions.check_model_params(params))
