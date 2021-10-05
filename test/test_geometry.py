import os
import unittest
import scipy.constants as con
from scipy.integrate import quad
from RaJePy.maths.physics import *

MSOL = 1.989e30
REL_ETOL = 1e-3  # Relative error tolerance as fraction of 'perfect' result

class TestPhysics(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls):
    #     pass
    #
    # @classmethod
    # def tearDownClass(cls):
    #     pass
    #
    # def setUp(self):
    #     pass
    #
    # def tearDown(self):
    #     pass

    def test_mod_r_0(self):
        pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
