"""
Example parameter file for the production of a physical jet model describing the
temperature, density, 3-D velocity and ionisation fraction within a 3-D grid.

Use would be with a RaJePy.classes.JetModel class instance e.g.:

jet_model = RaJePy.classes.JetModel('/full/path/to/example-model-params.py')
"""
import numpy as np

params = {
    "target": {"name": "test2",  # Jet/YSO/Model name
               "ra": "04:31:34.07736",  # HH:MM:SS.SS... [J2000]
               "dec": "+18:08:04.9020",  # DD:MM:SS.SS... [J2000]
               "epoch": "J2000",
               "dist": 140.,  # pc
               "v_lsr": 6.2,  # km/s, from Momose et al. (1998)
               "M_star": 0.55,  # M_sol in arcsec
               "R_1": .25,  # inner disc radii sourcing the jet in au
               "R_2": 2.5,  # outer disc radii sourcing the jet in au
               },
    "grid": {"n_x": 50,  # No. of cells in x
             "n_y": 400,  # No. of cells in y
             "n_z": 50,  # No. of cells in z
             "l_z": 0.5,  # Length of z-axis. Overrides n_x/n_y/n_z.
             "c_size": 0.5,  # Cell size (au)
             },
    "geometry": {"epsilon": 7. / 9.,  # Jet width index
                 "opang": 20.,  # Jet opening angle (deg)
                 "w_0": 2.5,  # Half-width of jet base (au)
                 "r_0": 2.5,  # Launching radius (au)
                 "inc": 90.,  # Inclination angle (deg)
                 "pa": 0.,  # Jet position PA (deg), 60deg nominally
                 "rotation": "CW",  # Rotation sense, one of CCW or CW
                 },
    "power_laws": {"q_v": 0.,  # Velocity index
                   "q_T": 0.,  # Temperature index
                   "q_x": 0.,  # Ionisation fraction index
                   "q^d_n": 0.,  # Cross-sectional density index
                   "q^d_T": 0.,  # Cross-sectional temperature index
                   "q^d_v": 0.,  # Cross-sectional velocity index
                   "q^d_x": 0.  # Cross-sectional ionisation fraction index
                   },
    "properties": {"v_0": 150.,  # Ejection velocity (km/s)
                   "x_0": 0.1,  # Initial HII fraction
                   "T_0": 1E4,  # Temperature (K)
                   "mu": 1.3,  # Mean atomic weight (m_H)
                   "mlr_bj": 1e-7,  # BLUE jet steady-state MLR [Msol/yr]
                   "mlr_rj": 5e-8,  # RED jet steady-state MLR [Msol/yr]
                   },
    "ejection": {"t_0": np.array([10.]),  # Peak times of bursts (yr)
                 "hl": np.array([0.5]),  # Half-lives of bursts (yr)
                 "chi": np.array([5.]),  # Burst factors
                 "which": np.array(["R", "B", "B", "RB"])  # Which jet the burst relates to
                 }
}
