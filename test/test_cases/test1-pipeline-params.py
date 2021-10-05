"""
Example parameter file for the running of a multi-frequency, multi-epoch
radiative transfer calculation and subsequent synthetic imaging with a variety
of telescopes/telescope-configurations.

Use would be with a RaJePy.classes.ModelRun class instance e.g.:

pipeline = RaJePy.classes.ModelRun('/full/path/to/example-pipeline-params.py')
"""

import os
import numpy as np


params = {'min_el':    20.,    # Min. elevation for synthetic observations (deg)
          'dcys':      {"model_dcy": os.sep.join([os.getcwd(), 'test_output_dcy'])},
          # Continuum observations
          'continuum': {'times':  np.array([0.]),  # yr
                        'freqs':  np.array([1.5, 3.0, 6., 10., 22., 33., 43., 5.05]) * 1e9,
                        't_obs':  np.array([1200, 1200, 3600, 1200, 1200, 1800, 2400, 59400]),
                        'tscps':  np.array([('VLA', 'A'), ('VLA', 'A'),
                                            ('VLA', 'A'), ('VLA', 'A'),
                                            ('VLA', 'A'), ('VLA', 'A'),
                                            ('VLA', 'A'), ('EMERLIN', '0')]),
                        't_ints': np.array([5, 5, 5, 5, 5, 5, 5, 5]),    # secs
                        'bws':    np.array([1e9, 2e9, 2e9, 4e9, 4e9, 4e9, 8e9, .5e9]),  # Hz
                        'chanws': np.array([1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8, 2.e8])},  # int
          # Radio recombination line observations
          'rrls':      {'times':  np.array([0.]),  # yr
                        'lines':  np.array(['H58a']),  # str (Element+n+dn)
                        't_obs':  np.array([30000]),  # secs
                        'tscps':  np.array([('VLA', 'A')]),  # (tscop, config)
                        't_ints': np.array([60]),  # secs
                        'bws':    np.array([1e8]),  # Hz
                        'chanws': np.array([1e5])},  # Hz
          }
# ############################################################################ #
