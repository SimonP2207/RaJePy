"""
Example parameter file for the running of a multi-frequency, multi-epoch
radiative transfer calculation and subsequent synthetic imaging with a variety
of telescopes/telescope-configurations.

Use would be with a RaJePy.classes.ModelRun class instance e.g.:

pipeline = RaJePy.classes.ModelRun('/full/path/to/example-pipeline-params.py')
"""

import os
import numpy as np

params = {'min_el':    20.,  # Minimum elevation for synthetic observations, deg
          'dcys':      {"model_dcy": os.sep.join([os.path.expanduser('~'), 'Desktop', 'test_output_rajepy'])},  # Output root directory
          # Continuum observations
          'continuum': {'times':  np.linspace(0., 5., 24 * 5 + 1),  # Model times, yr
                        'freqs':  np.array([6.]) * 1e9,  # Frequencies of observations, Hz
                        't_obs':  np.array([59400]),  # Total on-source times, s
                        'tscps':  np.array([('EMERLIN', '0')]),  # Array of 2-tuples of (telescope, configuration)
                        't_ints': np.array([5]),  # Visibility integration times, s
                        'bws':    np.array([.5e9]),  # Observational bandwidth, Hz
                        'chanws': np.array([2.e8])},  # Channel widths, Hz
          # Radio recombination line observations
          'rrls':      {'times':  np.array([]),  # Model times, yr
                        'lines':  np.array(['H58a']),  # RRL lines to observe (Element+n+dn)
                        't_obs':  np.array([30000]),  # Total on-source times, s
                        'tscps':  np.array([('VLA', 'A')]),  # List of 2-tuples of (telescope, configuration)
                        't_ints': np.array([60]),  # Visibility integration times, s
                        'bws':    np.array([1e8]),  # Observational bandwidth, Hz
                        'chanws': np.array([1e5])},  # Channel widths, Hz
          }
