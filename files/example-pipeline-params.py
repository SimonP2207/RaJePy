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
          'dcys':      {"model_dcy": os.sep.join([os.path.expanduser('~'),
                                                 "Desktop", "RaJePyTest"])},
          # Continuum observations
          'continuum': {'times':  np.linspace(0., 5., 21)[:1],  # yr
                        'freqs':  np.array([0.058, 0.142, 0.323, 0.608,  # Hz
                                            1.5, 3.0, 6., 10., 22., 33.,
                                            43.])[-1:] * 1e9,
                        't_obs':  np.array([28800, 28800, 28800, 28800,
                                            1200, 1200, 3600, 1200,
                                            1200, 1800, 2400])[6:7],
                        'tscps':  np.array([('LOFAR', '0'), ('LOFAR', '0'),
                                            ('GMRT', '0'), ('GMRT', '0'),
                                            ('VLA', 'A'), ('VLA', 'A'),
                                            ('VLA', 'A'), ('VLA', 'A'),
                                            ('VLA', 'A'), ('VLA', 'A'),
                                            ('VLA', 'A')])[6:7],
                        't_ints': np.array([5, 5, 5, 5, 5, 5,
                                            5, 5, 5, 5, 5])[6:7],    # secs
                        'bws':    np.array([30e6, 48e6, 32e6, 32e6,  # Hz
                                            1e9, 2e9, 2e9, 4e9, 4e9,
                                            4e9, 8e9])[6:7],  # Hz
                        'chanws': np.array([1e9] * 11)[6:7]},  # int
          # Radio recombination line observations
          'rrls':      {'times':  np.linspace(0., 5., 21)[:1],  # yr
                        'lines':  np.array(["H58a"]),  # str (Element+n+dn)
                        't_obs':  np.array([60 * 60 * 10]),  # secs
                        'tscps':  np.array([('VLA', 'A')]),  # (tscop, config)
                        't_ints': np.array([60]),  # secs
                        'bws':    np.array([128e6]),  # Hz
                        'chanws': np.array([1e6])},  # Hz
          }
# ############################################################################ #
if __name__ == '__main__':
    from RaJePy.miscellaneous import functions
    print(functions.check_pline_params(params))
