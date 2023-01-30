# -*- coding: utf-8 -*-
from typing import Callable, Union, Iterable
import os
import numpy as np
import scipy.constants as con
import pandas as pd
import dask.array as da
from uncertainties import ufloat as uf
from mpmath import gammainc
from . import geometry as geom
from ..physics import (approx_flux_expected_r86, flux_expected_r86,
                       flux_int_wrapped, mlr_from_n_0, n_0_from_mlr,
                       q_n, q_tau, r_tau1)
from ..physics import (atomic_mass, blackbody_nu, doppler_shift,
                       import_vanHoof2014, nu_rrl, rydberg_constant, z_number)
from ..physics import h_ss73, n_ss73, tau_ss73, temp_ss73, vr_ss73, u0_ss73
from RaJePy import cfg, cnsts


# ############################################################################ #
# ######################## Jet-model related methods below ################### #
# ############################################################################ #
def v_rot(r: Union[float, da.core.Array], reff: Union[float, da.core.Array],
          rho: Union[float, da.core.Array], epsilon: float,
          m_star: float) -> Union[float, da.core.Array]:
    """

    Parameters
    ----------
    r
        r-coordinate, au
    reff
        Effective radius (see RaJePy.maths.geometry.r_eff method), au
    rho
        Distance of r along r-axis in units of r_0 (see
        RaJePy.maths.geometry.mod_r_0 method)
    epsilon
        Power-law coefficient for jet-width as a function of distance along
        jet axis, r
    m_star
        Mass of central object around which Keplerian orbits are established
        (M_sol)
    Returns
    -------
    Rotational velocity in km/s
    """
    return da.sqrt(con.G * m_star * cnsts.MSOL / (reff * con.au)) * \
           rho ** -epsilon / 1e3


def tau_r(r: Union[float, da.core.Array], r_0: float, w_0: float, n_0: float,
          chi_0: float, T_0: float, freq: float, inc: float, epsilon: float,
          q_n: float, q_x: float, q_T: float,
          opang: float) -> Union[float, da.core.Array]:
    """
    Distance from central object to surface at which optical depth, tau,
    is unity (i.e. tau = 1). Equation 4 of Reynolds (1986).

    Parameters
    ----------
    r
        Radius at which to determine optical depth (in au)
    r_0
         Jet-launching radius (in au)
    w_0
         Half-width at r_0 (in au)
    n_0
         Number density at r_0 (in cm^-3)
    chi_0
         Ionisation fraction at r_0
    T_0
         Temperature at r_0 (in au)
    freq
        Frequency of observation (Hz)
    inc
         Inclination of jet to line of sight (in deg)
    epsilon
        Power-law exponent for jet-width as a function of r
    q_n
         Power-law for number density with distance along jet axis
         (dimensionless)
    q_x
         Power-law for ionisation fraction with distance along jet axis
         (dimensionless)
    q_T
         Power-law for temperature with distance along jet axis (dimensionless)
    opang
         Opening angle of jet (in deg)

    Returns
    -------
    Optical depth(s) of jet at distance r from central object

    """
    mr0 = geom.mod_r_0(opang, epsilon, w_0 * con.au * 1e2)
    q = epsilon + 2. * q_n + 2. * q_x - 1.35 * q_T
    tau = (2. * cnsts.a_k * (w_0 * con.au * 1e2) * n_0 ** 2. * chi_0 ** 2. *
           T_0 ** -1.35 *
           geom.rho(r * con.au * 1e2, r_0 * con.au * 1e2, mr0) ** q *
           freq ** -2.1 / np.sin(np.radians(inc)))

    return tau


def tau_r_from_jm(jm: 'JetModel', freq: float,
                  r: Union[float, da.core.Array]
                  ) -> Union[float, da.core.Array]:
    """
    Optical depth from Equations 4 + 5 of Reynolds (1986) analytical model
    paper, for monopolar jet. Parameters are extracted from a JetModel instance

    Parameters
    ----------
    jm
        Instance of JetModel class.
    freq
        Frequency of observation (Hz)
    r
        Distance along jet-axis at which to calculate tau (au)
    Returns
    -------
    Optical depth(s) of jet at distance r from central object
    """
    inc = jm.params['geometry']['inc']  # degrees
    r_0 = jm.params['geometry']['r_0']  # au
    opang = jm.params['geometry']['opang']  # deg
    w_0 = jm.params['geometry']['w_0']  # au
    d = jm.params['target']['dist']  # pc
    T_0 = jm.params['properties']['T_0']  # K
    n_0 = jm.params['properties']['n_0']  # cm^-3
    chi_0 = jm.params['properties']['x_0']  # dimensionless
    q_n = jm.params["power_laws"]["q_n"]  # dimensionless
    q_x = jm.params["power_laws"]["q_x"]  # dimensionless
    q_T = jm.params["power_laws"]["q_T"]  # dimensionless
    epsilon = jm.params["geometry"]["epsilon"]  # dimensionless

    return tau_r(r, r_0, w_0, n_0, chi_0, T_0, freq, inc,
                 epsilon, q_n, q_x, q_T, opang)


# ############################################################################ #
# ######################## Miscellaneous physics methods below ############### #
# ############################################################################ #
# TODO: Finish parallelising gff. You may have to parallelise import_vanHoof2014
#  too
def gff(freq, temp, z=1.):
    """
    Gaunt factors from van Hoof et al. (2014)
    """

    # Infinite-mass Rydberg unit of energy
    Ry = con.m_e * con.e ** 4. / (8 * con.epsilon_0 ** 2. * con.h ** 2.)

    logg2 = np.log10(z ** 2. * Ry / (con.k * temp))
    logu = np.log10(con.h * freq / (con.k * temp))

    from scipy.interpolate import interp2d
    logg2s, logus, gffs = import_vanHoof2014(errors=False)

    col = da.argmin(da.abs(logg2s[0] - logg2))
    row = da.argmin(da.abs(logus[:, 0] - logu))

    if col < 2:
        col = 2
    elif col > len(logg2s[0]) - 3:
        col = len(logg2s[0]) - 3
    if row < 2:
        row = 2
    elif row > len(logus[0]) - 3:
        row = len(logus[0]) - 3

    f = interp2d(logg2s[row - 2: row + 3, col - 2: col + 3],
                 logus[row - 2: row + 3, col - 2: col + 3],
                 gffs[row - 2: row + 3, col - 2: col + 3],
                 kind='cubic')

    return f(logg2, logu)
