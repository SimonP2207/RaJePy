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
                        'freqs':  np.array([15.]) * 1e9,
                        't_obs':  np.array([2400]),
                        'tscps':  np.array([('VLA', 'A')]),
                        't_ints': np.array([5]),    # secs
                        'bws':    np.array([2e9]),  # Hz
                        'chanws': np.array([1e8])},  # int
          # Radio recombination line observations
          'rrls':      {'times':  np.array([]),  # yr
                        'lines':  np.array([]),  # str (Element+n+dn)
                        't_obs':  np.array([]),  # secs
                        'tscps':  np.array([]),  # (tscop, config)
                        't_ints': np.array([]),  # secs
                        'bws':    np.array([]),  # Hz
                        'chanws': np.array([])},  # Hz
          }
# ############################################################################ #
