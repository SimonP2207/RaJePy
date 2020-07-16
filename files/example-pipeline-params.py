"""
Example parameter file for the running of a multi-frequency, multi-epoch
radiative transfer calculation and subsequent synthetic imaging with a variety
of telescopes/telescope-configurations.

Use would be with a VaJePy.classes.ModelRun class instance e.g.:

pipeline = VaJePy.classes.ModelRun('/full/path/to/example-pipeline-params.py')
"""

import os
import numpy as np

params = {'times': np.linspace(0., 5., 21),
          'freqs': np.array([0.058, 0.142, 0.323, 0.608, 1.5, 3.0, 6.,    # Hz
                             10., 22., 33., 44.]) * 1e9,
          't_obs': np.array([28800, 28800, 28800, 28800, 1200, 1200,     # secs
                             1200, 1200, 1200, 1800, 2400]),
          'tscps': np.array([('LOFAR', '0'), ('LOFAR', '0'),  # (tscop, config)
                             ('GMRT', '0'), ('GMRT', '0'), ('VLA', 'A'),
                             ('VLA', 'A'), ('VLA', 'A'), ('VLA', 'A'),
                             ('VLA', 'A'), ('VLA', 'A'), ('VLA', 'A')]),
          't_ints': np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]),    # secs
          'bws': np.array([30e6, 48e6, 32e6, 32e6, 1e9, 2e9, 2e9, 4e9, 4e9,
                           4e9, 8e9]),  # Hz
          'nchans': np.array([1] * 11),       # int
          'min_el': 20.,    # Minimum elevation for synthetic observations (deg)
          'dcys': {"model_dcy": os.sep.join([os.path.expanduser('~'),
                                             "Desktop", "RaJePyTest"])}}
