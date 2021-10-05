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
    #     None
    #
    # @classmethod
    # def tearDownClass(cls):
    #     None
    #
    # def setUp(self):
    #     None
    #
    # def tearDown(self):
    #     None

    def test_mlr_from_n_0(self):
        n0, mu, w0, v0 = 1e9, 1.3, 5.0, 300.
        qnd, qnv, r1, r2 = 0., 0., 0.5, 5.0

        r86_jmlr = con.pi * (n0 * 1e6) * (mu * atomic_mass("H")) * (w0 * con.au) ** 2. * (v0 * 1e3)
        r86_jmlr /= MSOL / con.year

        self.assertAlmostEqual(mlr_from_n_0(n0, v0, w0, mu, qnd, qnv, r1, r2), r86_jmlr,
                               delta=REL_ETOL * r86_jmlr)

        quad_constant = 2. * con.pi * n0 * mu * v0
        quad_constant *= 1e6 * 1e3 * con.au * atomic_mass("H")  # To SI units
        def quad_func(w, n0, v0, w0, mu, qnd, qnv, r1, r2):
            return quad_constant * (w * con.au) *\
                   (1. + w * (r2 - r1) / (w0 * r1)) ** (qnd + qnv)


        qnds, qnvs = (0., -0.5, -1.0, -1.5, -2.), (0., -0.5, -1.0, -1.5, -2.)

        for qnd in qnds:
            for qnv in qnvs:
                quad_result = quad(quad_func, 0., w0,
                                   args=(n0, v0, w0, mu,qnd, qnv, r1, r2))[0]
                quad_result *= con.year / MSOL
                result = mlr_from_n_0(n0, v0, w0, mu, qnd, qnv, r1, r2)
                self.assertAlmostEqual(result, quad_result,
                                       delta=REL_ETOL * quad_result)

    def test_n_0_from_mlr(self):
        mlr, mu, w0, v0 = 1e-6, 1.3, 5.0, 300.
        qnd, qnv, r1, r2 = 0., 0., 0.5, 5.0

        r86_n0 = (mlr * MSOL / con.year) / (con.pi * mu * atomic_mass('H') * (w0 * con.au) ** 2. * v0 * 1e3) / 1e6

        self.assertAlmostEqual(n_0_from_mlr(mlr, v0, w0, mu, qnd, qnv, r1, r2), r86_n0,
                               delta=REL_ETOL * r86_n0)


if __name__ == '__main__':
    unittest.main()
