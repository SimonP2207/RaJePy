import unittest
import numpy as np
import scipy.constants as con
from scipy.integrate import quad
import RaJePy.maths.physics as mphys

MSOL = 1.989e30
REL_ETOL = 1e-3  # Relative error tolerance as fraction of 'perfect' result


class TestPhysics(unittest.TestCase):
    QNDS = np.linspace(-2, 2, 9)
    QNVS = np.linspace(-2, 2, 9)

    def test_mlr_from_n_0(self):
        n0, mu, w0, v0, r1, r2 = 1e9, 1.3, 5.0, 300., 0.5, 5.0

        quad_constant = 2. * con.pi * n0 * mu * v0
        quad_constant *= 1e6 * 1e3 * mphys.atomic_mass("H")  # To SI units

        # Define integral
        def quad_func(w, w0_, qnd_, qnv_, r1_, r2_):
            return w * (1. + w * (r2_ - r1_) / (w0_ * r1_)) ** (qnd_ + qnv_)

        for qnd__ in self.QNDS:
            for qnv__ in self.QNVS:
                quad_result = quad(quad_func, 0., w0 * con.au,
                                   args=(w0 * con.au, qnd__, qnv__,
                                         r1 * con.au, r2 * con.au))[0]
                comparison_result = quad_result * quad_constant
                comparison_result /= MSOL / con.year
                result = mphys.mlr_from_n_0(n0, v0, w0, mu, qnd__, qnv__,
                                            r1, r2)
                self.assertAlmostEqual(result, comparison_result,
                                       delta=REL_ETOL * comparison_result)

    def test_n_0_from_mlr(self):
        mlr, mu, w0, v0, r1, r2, = 1e-6, 1.3, 5.0, 400., 0.5, 5.0

        quad_constant = 2. * con.pi * mu * v0
        quad_constant *= 1e3 * mphys.atomic_mass("H")  # To SI units

        # Define integral
        def quad_func(w, w0_, qnd_, qnv_, r1_, r2_):
            return w * (1. + w * (r2_ - r1_) / (w0_ * r1_)) ** (qnd_ + qnv_)

        for qnd__ in self.QNDS:
            for qnv__ in self.QNVS:
                quad_result = quad(quad_func, 0., w0 * con.au,
                                   args=(w0 * con.au, qnd__, qnv__,
                                         r1 * con.au, r2 * con.au))[0]
                comparison_result = ((mlr * MSOL / con.year) /
                                     (quad_result * quad_constant)) * 1e-6
                result = mphys.n_0_from_mlr(mlr, v0, w0, mu, qnd__, qnv__,
                                            r1, r2)
                self.assertAlmostEqual(result, comparison_result,
                                       delta=REL_ETOL * comparison_result)


if __name__ == '__main__':
    unittest.main()
