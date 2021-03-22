# -*- coding: utf-8 -*-
from typing import Callable
import os
import numpy as np
import scipy.constants as con
import pandas as pd
from scipy.integrate import tplquad
from mpmath import gammainc
from RaJePy import _config as cfg
from RaJePy import _constants as cnsts
from RaJePy import classes as clss
from RaJePy.maths import geometry as geom
from uncertainties import ufloat as uf

def tau_r(r, r_0, w_0, n_0, chi_0, T_0, freq, inc, epsilon, q_n, q_x, q_T,
          opang, dist=None):
    """
    Distance from central object to surface at which optical depth, tau,
    is equal to 1. Equation 4 of Reynolds (1986).

    Parameters
    ----------
    r : float
        Radius at which to determine optical depth (in au)
    r_0 : float
         Jet-launching radius (in au)
    w_0 : float
         Half-width at r_0 (in au)
    n_0 : float
         Number density at r_0 (in cm^-3)
    chi_0 : float
         Ionisation fraction at r_0
    T_0 : float
         Temperature at r_0 (in au)
    freq : float
        Frequency of observation (Hz)
    inc : float
         Inclination of jet to line of sight (in deg)
    q_n : float
         Power-law for number density with distance along jet axis
         (dimensionless)
    q_x : float
         Power-law for ionisation fraction with distance along jet axis
         (dimensionless)
    q_T : float
         Power-law for temperature with distance along jet axis (dimensionless)
    opang : float
         Opening angle of jet (in deg)
    dist: float
        Distance to target (pc). Default is None.

    Returns
    -------
    float
        Distance to tau = 1 surface from central object. If arg dist is None,
        in au, if arg dist in pc, in arcsec

    """
    a_k = 0.212  # given as constants of cgs equations
    m_r_0 = geom.mod_r_0(opang, epsilon, w_0 * con.au * 1e2)
    q = epsilon + 2. * q_n + 2. * q_x - 1.35 * q_T
    rho = ((r * con.au * 1e2) + m_r_0 - (r_0 * con.au * 1e2)) / m_r_0
    tau = (2. * a_k * (w_0 * con.au * 1e2) * n_0 ** 2. * chi_0 ** 2. *
           T_0 ** -1.35 * rho ** q * freq ** -2.1 / np.sin(np.radians(inc)))

    return tau

def r_tau1(r_0, w_0, n_0, chi_0, T_0, freq, inc, epsilon, q_n, q_x, q_T, opang,
           dist=None):
    """
    Distance from central object to surface at which optical depth, tau,
    is equal to 1. Equation 4 of Reynolds (1986).

    Parameters
    ----------
    r_0 : float
         Jet-launching radius (in au)
    w_0 : float
         Half-width at r_0 (in au)
    n_0 : float
         Number density at r_0 (in cm^-3)
    chi_0 : float
         Ionisation fraction at r_0
    T_0 : float
         Temperature at r_0 (in au)
    freq : float
        Frequency of observation (Hz)
    inc : float
         Inclination of jet to line of sight (in deg)
    q_n : float
         Power-law for number density with distance along jet axis
         (dimensionless)
    q_x : float
         Power-law for ionisation fraction with distance along jet axis
         (dimensionless)
    q_T : float
         Power-law for temperature with distance along jet axis (dimensionless)
    opang : float
         Opening angle of jet (in deg)
    dist: float
        Distance to target (pc). Default is None.

    Returns
    -------
    float
        Distance to tau = 1 surface from central object. If arg dist is None,
        in au, if arg dist in pc, in arcsec

    """
    a_j, a_k = 6.5E-38, 0.212  # given as constants of cgs equations
    m_r_0 = geom.mod_r_0(opang, epsilon, w_0 * con.au * 1e2)
    q = epsilon + 2. * q_n + 2. * q_x - 1.35 * q_T
    rho = (2. * a_k * (w_0 * con.au * 1e2) * n_0 ** 2. * chi_0 ** 2. *
           T_0 ** -1.35 * freq ** -2.1 * np.sin(np.radians(inc))
           ** -1.) ** (-1. / q)

    r = rho * m_r_0 + (r_0 * con.au * 1e2) - m_r_0

    if dist is None:
        return r

    return r / (con.au * 1e2) / dist

def z_number(atom: str) -> int:
    """
    Atomic number of atom

    :param str atom: Atomic nucleus' element as per periodic table symbol
    :return: Z
    :rtype: int
    """
    return {'H': 1, 'He': 2, 'Li': 3, 'Be': 4,
            'B': 5, 'C': 6, 'N': 7, 'O': 8}[atom]


def rydberg_constant(atom: str):
    """
    Return the Rydberg constant for relevant nucleus.

    :param str atom: Atomic nucleus' element as per periodic table symbol
    :return: Rydberg constant (m\ :sup:`-1`)
    :rtype: float
    """
    m_atom = atomic_mass(atom)
    return con.Rydberg * (m_atom / (m_atom + con.m_e))


def doppler_shift(nu_0: float, v_lsr: float) -> float:
    """
    Doppler shifted line rest frequency. Equation 2.17 of Gordon & Sorochenko

    :param float nu_0: Non-Doppler-shifted rest frequency (Hz)
    :param float v_lsr: System velocity with respect to local standard of
                        rest (km/s)
    :return: :math:`\nu_{0, Doppler}` (Hz)
    :rtype: float
    """
    v = v_lsr * 1000.  # km/s to m/s
    return nu_0 * (1. - v / con.c)


def blackbody_nu(freq: float, temp: float) -> float:
    """
    Planck's law as function of frequency returning the spectal radiance of
    a black body at temperature, T, at a frequency, freq.

    :param float freq: Frequency of observation (Hz)
    :param float temp: Electron temperature (K)
    :return: B_nu(nu,T) (erg s^-1 cm^-2 Hz^-1 sr^-1)
    :rtype: float
    """
    p1 = 2. * con.h * 1e7 * freq ** 3. / (con.c * 1e2) ** 2.
    p2 = np.exp(con.h * 1e7 * freq / (con.k * 1e7 * temp)) - 1.

    return p1 * p2 ** -1.


def nu_rrl(n, dn=1, atom="H"):
    """
    Calculate radio recombination line frequency(s)

    Parameters
    ----------
    n : int, float or Iterable
        Electronic transition number.

    dn : int
        Number of levels transitioned e.g. for alpha RRLs dn = 1

    atom : str
        Chemical symbol for atom to compute RRLs for e.g. 'H' for Hydrogen,
        'He' for Helium etc. Available for Hydrogen up to Magnesium.
    Returns
    -------
    float or Iterable
        DESCRIPTION.

    """
    n_p, n_n = cnsts.NZ[atom]

    M = atomic_mass(atom)
    M -= con.m_e * n_p

    R_M = con.Rydberg * (1. + con.m_e / M) ** -1.
    return R_M * con.c * (1. / n ** 2. - 1. / (n + dn) ** 2.)


def atomic_mass(atom):
    """
    Calculate mass of atom

    Parameters
    ----------
    atom : str
        Chemical symbol to return mass of e.g. "H", "He" etc.

    Returns
    -------
    Mass of atom in kg.
    """
    ams = pd.read_pickle(cfg.dcys["files"] + os.sep + "atomic_masses.pkl")
    n_p, n_n = cnsts.NZ[atom]
    M = ams[(ams['N'] == n_n) & (ams['Z'] == n_p)]['mass[micro-u]'].values[0]
    M *= 1e-6 * con.u
    return M


def approx_flux_expected_r86(jm, freq):
    """
    Approximate flux expected from Equation 16 of Reynolds (1986) analytical
    model paper, for monopolar jet.

    Parameters
    ----------
    jm : JetModel
        Instance of JetModel class.
    freq : float or Iterable
        Frequency of observation (Hz)

    Returns
    -------
    float or Iterable
        Flux (Jy).

    """
    if type(freq) == list:
        freq = np.array(freq)

    w_0 = jm.params['geometry']['w_0'] * con.au * 1e2

    # Assume even density across jet if disc-wind prescription adopted
    if jm.params["power_laws"]["q^d_n"] != 0.:
        mlr = jm.params["properties"]["mlr"] * 1.989e30 / con.year
        n_0 = mlr / (np.pi * jm.params['properties']['mu'] * atomic_mass("H") *
                     w_0 ** 2. * jm.params['properties']["v_0"] * 1e5)
    else:
        n_0 = jm.params['properties']['n_0']

    a_j, a_k = 6.5E-38, 0.212  # given as constants of cgs equations
    c = (1. + jm.params['geometry']['epsilon'] +
         jm.params['power_laws']['q_T']) / jm.params['power_laws']['q_tau']
    flux = 2 ** (1. - c) * (
                jm.params['target']['dist'] * con.parsec * 1e2) ** -2.
    flux *= a_j * a_k ** (-1. - c) * jm.params['properties']['T_0'] ** (
                1. + 1.35 * c)
    flux *= jm.params['geometry']['mod_r_0'] * con.au * 1e2
    flux *= w_0 ** (1. - c)
    flux *= (n_0 * jm.params['properties']['x_0']) ** (-(2. * c))
    flux *= np.sin(np.radians(jm.params['geometry']['inc'])) ** (1. + c) / \
            (c * (1. + jm.params['geometry']['epsilon'] +
                  jm.params['power_laws']['q_T'] +
                  jm.params['power_laws']['q_tau']))
    alpha = 2. + (2.1 / jm.params['power_laws']['q_tau']) * \
            (1 + jm.params['geometry']['epsilon'] +
             jm.params['power_laws']['q_T'])

    flux *= freq ** alpha  # in erg cm^-2 s^-1 Hz^-1
    flux *= 1e-7 * 1e2 ** 2.  # now in W m^-2 Hz^-1
    return flux / 1e-26  # now in Jy


def flux_expected_r86(jm, freq, y_max, y_min=None):
    """
    Exact flux expected from Equation 8 of Reynolds (1986) analytical model
    paper, for monopolar jet.

    Parameters
    ----------
    jm : JetModel
        Instance of JetModel class.
    freq : float
        Frequency of observation (Hz)
    y_max : float
        Jet's angular extent to integrate flux over (arcsecs).
    y_min : float
        Minimum value from jet base to integrate from (arcsecs)
    Returns
    -------
    float
        Exact flux expected from Reynolds (1986)'s analytical model (Jy).

    """
    # Parse constants into local variables
    a_j, a_k = 6.5E-38, 0.212  # given as constants of cgs equations
    inc = jm.params['geometry']['inc']  # degrees
    w_0 = jm.params['geometry']['w_0'] * con.au * 1e2  # cm
    T_0 = jm.params['properties']['T_0']  # K
    n_0 = jm.params['properties']['n_0']  # cm^-3
    x_0 = jm.params['properties']['x_0']  # dimensionless
    q_tau = jm.params["power_laws"]["q_tau"]  # dimensionless
    q_T = jm.params["power_laws"]["q_T"]  # dimensionless
    eps = jm.params["geometry"]["epsilon"]  # dimensionless
    mod_r_0 = jm.params['geometry']['mod_r_0'] * con.au * 1e2  # cm
    mod_y_0 = mod_r_0 * np.sin(np.radians(inc))  # cm
    r_0 = jm.params['geometry']['r_0'] * con.au * 1e2  # cm
    y_0 = r_0 * np.sin(np.radians(inc))  # cm

    d = jm.params['target']['dist'] * con.parsec * 1e2  # cm

    # Assume even density across jet if disc-wind prescription adopted
    if jm.params["power_laws"]["q^d_n"] != 0.:
        mlr = jm.params["properties"]["mlr"] * 1.989e30 / con.year
        n_0 = mlr / (np.pi * jm.params['properties']['mu'] * atomic_mass("H") *
                     w_0 ** 2. * jm.params['properties']["v_0"] * 1e5)

    # Convert y_max/y_min from arcseconds to cm
    y_max = np.tan(y_max * con.arcsec) * d + mod_y_0 - y_0  # in cm
    if y_min is not None:
        y_min = np.tan(y_min * con.arcsec) * d + mod_y_0 - y_0  # in cm
    else:
        y_min = mod_y_0

    # Calculate optical depth at base of jet
    tau_0 = 2. * a_k * w_0 * (n_0 * x_0) ** 2. * T_0 ** -1.35 * freq ** -2.1 * \
            np.sin(np.radians(inc)) ** -1.

    c = 1. + eps + q_T

    def indef_integral(yval):
        """yval in cm"""
        const = 2. * w_0 * d ** -2. * a_j * a_k ** -1. * T_0 * freq ** 2.
        rho = yval / mod_y_0
        tau = tau_0 * rho ** q_tau

        p1 = yval / (q_tau * c) * rho ** (c - 1.) * tau ** (-c / q_tau)
        p2 = q_tau * tau ** (c / q_tau) + c * gammainc(c / q_tau, tau)

        return const * (float(p1) * float(p2))

    flux = indef_integral(y_max) - indef_integral(y_min)
    flux *= 1e-7 * 1e2 ** 2.  # W m^-2 Hz^-1

    return flux / 1e-26


# def flux_expected_r86_cs(jm, freq, y_max, y_min=None):
#     """
#     Exact flux expected from Equation 8 of Reynolds (1986) analytical model
#     paper, for monopolar jet.
#
#     Parameters
#     ----------
#     jm : JetModel
#         Instance of JetModel class.
#     freq : float
#         Frequency of observation (Hz)
#     y_max : float
#         Jet's angular extent to integrate flux over (arcsecs).
#     y_min : float
#         Minimum value from jet base to integrate from (arcsecs)
#     Returns
#     -------
#     float
#         Exact flux expected from Reynolds (1986)'s analytical model (Jy).
#
#     """
#     # Parse constants into local variables
#     a_j, a_k = 6.5E-38, 0.212  # given as constants of cgs equations
#     inc = jm.params['geometry']['inc']  # degrees
#     pa = 0.  # degrees
#     w_0 = jm.params['geometry']['w_0'] * con.au * 1e2  # cm
#     T_0 = jm.params['properties']['T_0']  # K
#     n_0 = jm.params['properties']['n_0']  # cm^-3
#     x_0 = jm.params['properties']['x_0']  # dimensionless
#     q_tau = jm.params["power_laws"]["q_tau"]  # dimensionless
#     q_T = jm.params["power_laws"]["q_T"]  # dimensionless
#     eps = jm.params["geometry"]["epsilon"]  # dimensionless
#     mod_r_0 = jm.params['geometry']['mod_r_0'] * con.au * 1e2  # cm
#     mod_y_0 = mod_r_0 * np.sin(np.radians(inc))  # cm
#     r_0 = jm.params['geometry']['r_0'] * con.au * 1e2  # cm
#     y_0 = r_0 * np.sin(np.radians(inc))  # cm
#     R_1 = jm.params["target"]["R_1"]
#     R_2 = jm.params["target"]["R_2"]
#     dist = jm.params["target"]["dist"] * con.parsec * 1e2  # cm
#
#     q_n = jm.params["power_laws"]["q_n"]
#     q_x = jm.params["power_laws"]["q_x"]
#     q_T = jm.params["power_laws"]["q_T"]
#     q_nd = jm.params["power_laws"]["q^d_n"]
#     q_xd = jm.params["power_laws"]["q^d_x"]
#     q_Td = jm.params["power_laws"]["q^d_T"]
#
#     # def tau_xyz(x, z):
#     # def tau_xyz(x, z):
#     #     def func2(y):
#     #         r, w = geom.xyz_to_rw(x, y, z, inc, 0.)
#     #         wr = geom.w_r(r, w_0, mod_r_0, r_0, eps)
#     #
#     #         if w > wr or r < r_0:
#     #             return 0.
#     #
#     #         const = a_k * n_0 ** 2. * x_0 ** 2. * T_0 ** -1.35
#     #         const *= freq ** -2.1
#     #
#     #         expnt1 = q_n * 2. + q_x * 2. - 1.35 * q_T
#     #         rho1 = ((r + mod_r_0 - r_0) / mod_r_0) ** expnt1
#     #
#     #         expnt2 = q_nd * 2. + q_xd * 2. - 1.35 * q_Td
#     #         rho2 = (geom.r_eff(w, R_1, R_2, w_0, r, mod_r_0, r_0, eps) /
#     #                 R_1) ** expnt2
#     #
#     #         return const * rho1 * rho2
#     #     return func2
#     # a, b = geom.y1_y2(x, z, w_0, r_0, mod_r_0, inc)
#     # return quad(func1(x, z), a, b)
#
#     # Lower and upper bounds in y
#     qfun = geom.y1_y2_wrapped(w_0, r_0, mod_r_0, inc, bound='lower')
#     rfun = geom.y1_y2_wrapped(w_0, r_0, mod_r_0, inc, bound='upper')
#
#     # Lower and upper bounds in x
#     gfun = geom.w_r_wrapped(w_0, mod_r_0, r_0, eps, inc, bound='lower')
#     hfun = geom.w_r_wrapped(w_0, mod_r_0, r_0, eps, inc, bound='upper')
#
#     # Lower and upper bounds in z
#     a = r_0 * np.sin(np.radians(inc))
#     b = np.tan(y_max * con.arcsec) * dist
#
#     return tplquad(flux_int_wrapped(freq, jm), a, b, gfun, hfun, qfun, rfun)


def flux_int_wrapped(freq: float, jm) -> Callable:
    # Parse constants into local variables
    a_j, a_k = 6.5E-38, 0.212  # given as constants of cgs equations
    inc = jm.params['geometry']['inc']  # degrees
    w_0 = jm.params['geometry']['w_0'] * con.au * 1e2  # cm
    T_0 = jm.params['properties']['T_0']  # K
    n_0 = jm.params['properties']['n_0']  # cm^-3
    x_0 = jm.params['properties']['x_0']  # dimensionless
    q_T = jm.params["power_laws"]["q_T"]  # dimensionless
    eps = jm.params["geometry"]["epsilon"]  # dimensionless
    mod_r_0 = jm.params['geometry']['mod_r_0'] * con.au * 1e2  # cm
    r_0 = jm.params['geometry']['r_0'] * con.au * 1e2  # cm
    R_1 = jm.params["target"]["R_1"]
    R_2 = jm.params["target"]["R_2"]
    dist = jm.params["target"]["dist"] * con.parsec * 1e2  # cm

    q_n = jm.params["power_laws"]["q_n"]
    q_x = jm.params["power_laws"]["q_x"]
    q_T = jm.params["power_laws"]["q_T"]
    q_nd = jm.params["power_laws"]["q^d_n"]
    q_xd = jm.params["power_laws"]["q^d_x"]
    q_Td = jm.params["power_laws"]["q^d_T"]

    def func(z, x, y):
        r, w, _ = geom.xyz_to_rwp(x, y, z, inc, 0.)
        wr = geom.w_r(r, w_0, mod_r_0, r_0, eps)
        if w > wr or r < r_0:
            # s = 'help: x={:.2f} y={:.2f} z={:.2f} r={:.2f} w={:.2f}'
            # print(s.format(*[_ / con.au / 1e2 for _ in [x, y, z, r, w]]))
            return 0.
        const_tau = a_k * n_0 ** 2. * x_0 ** 2. * T_0 ** -1.35
        const_tau *= freq ** -2.1

        expnt1_tau = q_n * 2. + q_x * 2. - 1.35 * q_T
        rho1_tau = ((r + mod_r_0 - r_0) / mod_r_0) ** expnt1_tau

        expnt2_tau = q_nd * 2. + q_xd * 2. - 1.35 * q_Td
        rho2_tau = (geom.r_eff(w, R_1, R_2, w_0, r, mod_r_0, r_0, eps) /
                    R_1) ** expnt2_tau
        tau = const_tau * rho1_tau * rho2_tau

        rho1_T = ((r + mod_r_0 - r_0) / mod_r_0) ** q_T
        rho2_T = (geom.r_eff(w, R_1, R_2, w_0, r, mod_r_0, r_0, eps) /
                  R_1) ** q_Td
        temp = T_0 * rho1_T * rho2_T

        return dist ** -2. * a_j / a_k * temp * freq ** 2. * (1. - np.exp(-tau))

    return func


def import_vanHoof2014(errors=False):
    datafile = os.sep.join([cfg.dcys['files'], "vanHoofetal2014.data"])

    data = []
    with open(datafile, 'rt') as f:
        line_count = 0
        lines = f.readlines()
        loggam2_start = float(lines[30].split('#')[0])
        logu_start = float(lines[31].split('#')[0])
        step = float(lines[32].split('#')[0])

        # Calculated values for g_ff
        data_lines = [[float(_) for _ in l.split()] for l in lines[42:188]]
        n_logu = len(data_lines)
        n_loggamma2 = len(data_lines[0])

        # Uncertainties in calculated values for g_ff
        unc_lines = [[float(_) for _ in l.split()] for l in lines[192:]]

        logus = np.linspace(np.round(logu_start, decimals=1),
                            np.round(logu_start + (step * (n_logu - 1)),
                                     decimals=1), n_logu)

        loggam2s = np.linspace(np.round(loggam2_start, decimals=1),
                               np.round(loggam2_start +
                                        (step * (n_loggamma2 - 1)),
                                        decimals=1),
                               n_loggamma2)

        loggam2s, logus = np.meshgrid(loggam2s, logus)
        gffs = np.zeros(np.shape(loggam2s))

        for idx1, line in enumerate(data_lines):
            for idx2, gff in enumerate(line):
                if errors:
                    gffs[idx1][idx2] = uf(gff, unc_lines[idx1][idx2])
                else:
                    gffs[idx1][idx2] = gff

    return loggam2s, logus, np.array(gffs)


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

    col = np.argmin(np.abs(logg2s[0] - logg2))
    row = np.argmin(np.abs(logus[:, 0] - logu))

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


gff = np.vectorize(gff)


def tau_r_from_jm(jm, freq, r) -> float:
    """
    Optical depth from Equations 4 + 5 of Reynolds (1986) analytical model
    paper, for monopolar jet. Parameters are extracted from a JetModel instance

    Parameters
    ----------
    jm : JetModel
        Instance of JetModel class.
    freq : float
        Frequency of observation (Hz)
    r: float
        Distance along jet-axis at which to calculate tau (au)
    Returns
    -------
    float
        Distance to tau = 1 surface from central object in arcsec
    """
    inc = jm.params['geometry']['inc']  # degrees
    r_0 = jm.params['geometry']['r_0'] * con.au * 1e2  # cm
    opang = jm.params['geometry']['opang']  # deg
    w_0 = jm.params['geometry']['w_0'] * con.au * 1e2  # cm
    d = jm.params['target']['dist'] * con.parsec * 1e2  # cm
    T_0 = jm.params['properties']['T_0']  # K
    n_0 = jm.params['properties']['n_0']  # cm^-3
    chi_0 = jm.params['properties']['x_0']  # dimensionless
    q_n = jm.params["power_laws"]["q_n"]  # dimensionless
    q_x = jm.params["power_laws"]["q_x"]  # dimensionless
    q_T = jm.params["power_laws"]["q_T"]  # dimensionless
    epsilon = jm.params["power_laws"]["epsilon"]

    return tau_r(r, r_0, w_0, n_0, chi_0, T_0, freq, inc, epsilon, q_n, q_x,
                 q_T, opang, dist=d)


if __name__ == '__main__':
    from RaJePy.classes import JetModel
    import RaJePy as rjp
    import matplotlib.pylab as plt

    jm = JetModel(rjp.cfg.dcys['files'] + os.sep + 'example-model-params.py')

    freqs = np.logspace(8, 12, 13)
    fluxes, fluxes_approx = [], []
    r_0 = jm.params['geometry']['r_0']
    dist = jm.params['target']['dist']
    for freq in freqs:
        fluxes.append(2 * flux_expected_r86(jm, freq, 10.))
        fluxes_approx.append(2. * approx_flux_expected_r86(jm, freq))

    plt.close('all')

    plt.loglog(freqs, fluxes, 'ro')
    plt.loglog(freqs, fluxes_approx, 'gx')

    plt.show()
    #
