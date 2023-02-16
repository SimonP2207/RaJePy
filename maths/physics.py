# -*- coding: utf-8 -*-
import os
from typing import Callable, Union, Iterable

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.constants as con
from uncertainties import ufloat as uf


# ############################################################################ #
# ######################## Jet-model related methods below ################### #
# ############################################################################ #
def q_n(epsilon: float, q_v: float) -> float:
    """
    Power-law coefficient for number density as a function of distance along
    jet axis, r. This calculation conserves mass along jet (see Reynolds, 1986)

    Parameters
    ----------
    epsilon
        Power-law coefficient for jet-width as a function of distance along
        jet axis, r
    q_v
        Power-law coefficient for velocity as a function of distance along
        jet axis, r

    Returns
    -------
    Power-law coefficient for number-density along jet axis
    """
    return -q_v - (2.0 * epsilon)


def q_tau(epsilon: float, q_x: float, q_n: float, q_T: float) -> float:
    """
    Power-law coefficient for optical depth as a function of distance along
    jet axis, r (see Reynolds, 1986).

    Parameters
    ----------
    epsilon
        Power-law coefficient for jet-width as a function of distance along
        jet axis, r
    q_x
        Power-law coefficient for ionisation fraction as a function of distance
        along jet axis, r
    q_n
        Power-law coefficient for number density as a function of distance along
        jet axis, r
    q_T
        Power-law coefficient for temperature as a function of distance along
        jet axis, r

    Returns
    -------
    Power-law coefficient for optical depth along jet axis
    """

    return epsilon + 2.0 * q_x + 2.0 * q_n - 1.35 * q_T


def v_rot(reff: Union[float, Iterable],
          rho: Union[float, Iterable], epsilon: float,
          m_star: float) -> Union[float, Iterable]:
    """

    Parameters
    ----------
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
    from .. import cnsts
    
    return np.sqrt(con.G * m_star * cnsts.MSOL /
                   (reff * con.au)) * rho ** -epsilon / 1e3


def tau_r(r, r_0, w_0, n_0, chi_0, T_0, freq, inc, epsilon, q_n, q_x, q_T,
          opang):
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
    epsilon : float
        Power-law exponent for jet-width as a function of r
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

    Returns
    -------
    float
        Optical depth of jet at distance r from central object

    """
    from .. import cnsts
    from . import geometry as geom

    mr0 = geom.mod_r_0(opang, epsilon, w_0 * con.au * 1e2)
    q = epsilon + 2. * q_n + 2. * q_x - 1.35 * q_T
    tau = (2. * cnsts.a_k * (w_0 * con.au * 1e2) * n_0 ** 2. * chi_0 ** 2. *
           T_0 ** -1.35 *
           geom.rho(r * con.au * 1e2, r_0 * con.au * 1e2, mr0) ** q *
           freq ** -2.1 / np.sin(np.radians(inc)))

    return tau


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
    epsilon : float
        Power-law exponent for jet-width as a function of r
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
    from .. import cnsts
    from . import geometry as geom

    m_r_0 = geom.mod_r_0(opang, epsilon, w_0 * con.au * 1e2)
    q = epsilon + 2. * q_n + 2. * q_x - 1.35 * q_T
    rho = (2. * cnsts.a_k * (w_0 * con.au * 1e2) * n_0 ** 2. * chi_0 ** 2. *
           T_0 ** -1.35 * freq ** -2.1 * np.sin(np.radians(inc))
           ** -1.) ** (-1. / q)

    r = rho * m_r_0 + (r_0 * con.au * 1e2) - m_r_0

    if dist is None:
        return r

    return r / (con.au * 1e2) / dist


def approx_flux_expected_r86(jm: 'JetModel', freq: float, which: str):
    """
    Approximate flux expected from Equation 16 of Reynolds (1986) analytical
    model paper, for monopolar jet.

    Parameters
    ----------
    jm
        Instance of JetModel class.
    freq
        Frequency of observation (Hz)
    which
        Red or blue jet? 'R' or 'B', respectively

    Returns
    -------
    float or Iterable
        Flux (Jy).

    """
    from .. import cnsts

    if type(freq) == list:
        freq = np.array(freq)

    w_0 = jm.params['geometry']['w_0'] * con.au * 1e2

    # Assume even density across jet if disc-wind prescription adopted
    if jm.params["power_laws"]["q^d_n"] != 0.:
        mlr = jm.ss_jml(which)
        n_0 = mlr / (np.pi * jm.params['properties']['mu'] * atomic_mass("H") *
                     w_0 ** 2. * jm.params['properties']["v_0"] * 1e5)
    else:
        n_0 = jm.params['properties']['n_0']

    if which == 'R':
        n_0 *= jm.ss_jml('R') / jm.ss_jml('B')

    c = (1. + jm.params['geometry']['epsilon'] +
         jm.params['power_laws']['q_T']) / jm.params['power_laws']['q_tau']
    flux = 2 ** (1. - c) * (
            jm.params['target']['dist'] * con.parsec * 1e2) ** -2.
    flux *= cnsts.a_j * cnsts.a_k ** (-1. - c) *\
            jm.params['properties']['T_0'] ** (1. + 1.35 * c)
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


def flux_expected_r86(jm, freq, which: str, y_max, y_min=None):
    """
    Exact flux expected from Equation 8 of Reynolds (1986) analytical model
    paper, for monopolar jet.

    Parameters
    ----------
    jm : JetModel
        Instance of JetModel class.
    freq : float
        Frequency of observation (Hz)
    which
        Red or blue jet? 'R' or 'B', respectively
    y_max : float
        Jet's angular extent to integrate flux over (arcsecs).
    y_min : float
        Minimum value from jet base to integrate from (arcsecs)
    Returns
    -------
    float
        Exact flux expected from Reynolds (1986)'s analytical model (Jy).

    """
    from mpmath import gammainc
    from .. import cnsts

    # Parse constants into local variables
    inc = jm.params['geometry']['inc']  # degrees
    w_0 = jm.params['geometry']['w_0'] * con.au * 1e2  # cm
    T_0 = jm.params['properties']['T_0']  # K
    n_0 = jm.params['properties']['n_0']  # cm^-3

    if which == 'R':
        n_0 *= jm.ss_jml('R') / jm.ss_jml('B')

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
        mlr = jm.ss_jml(which)
        n_0 = mlr / (np.pi * jm.params['properties']['mu'] * atomic_mass("H") *
                     w_0 ** 2. * jm.params['properties']["v_0"] * 1e5)

    # Convert y_max/y_min from arcseconds to cm
    y_max = np.tan(y_max * con.arcsec) * d + mod_y_0 - y_0  # in cm
    if y_min is not None:
        y_min = np.tan(y_min * con.arcsec) * d + mod_y_0 - y_0  # in cm
    else:
        y_min = mod_y_0

    # Calculate optical depth at base of jet
    tau_0 = 2. * cnsts.a_k * w_0 * (n_0 * x_0) ** 2. * T_0 ** -1.35 * \
            freq ** -2.1 * np.sin(np.radians(inc)) ** -1.

    c = 1. + eps + q_T

    def indef_integral(yval):
        """yval in cm"""
        const = 2. * w_0 * d ** -2. * cnsts.a_j * cnsts.a_k ** -1. * T_0 *\
                freq ** 2.
        rho = yval / mod_y_0
        tau = tau_0 * rho ** q_tau

        p1 = yval / (q_tau * c) * rho ** (c - 1.) * tau ** (-c / q_tau)
        p2 = q_tau * tau ** (c / q_tau) + c * gammainc(c / q_tau, tau)

        return const * (float(p1) * float(p2))

    flux = indef_integral(y_max) - indef_integral(y_min)
    flux *= 1e-7 * 1e2 ** 2.  # W m^-2 Hz^-1

    return flux / 1e-26


def flux_int_wrapped(freq: float, jm) -> Callable:
    from .. import cnsts
    from . import geometry as geom

    # Parse constants into local variables
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
        const_tau = cnsts.a_k * n_0 ** 2. * x_0 ** 2. * T_0 ** -1.35
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

        return dist ** -2. * cnsts.a_j / cnsts.a_k * temp * freq ** 2. *\
               (1. - np.exp(-tau))

    return func


def mlr_from_n_0(n_0: float, v_0: float, w_0: float, mu: float, q_nd: float,
                 q_nv: float, R_1: float, R_2: float) -> float:
    """
    Calculate jet mass loss rate, given central number density at the base of
    the jet

    Parameters
    ----------
    n_0 : float
        Central number density at jet-base (cm^-3)
    v_0 : float
        Central velocity at the base of the jet (km / s)
    w_0 : float
        Full jet-width at the base of teh jet (au)
    mu : float
        Average atomic mass (u)
    q_nd : float
        Power-law exponent for number density as a function of w
    q_nv : float
        Power-law exponent for velocity as a function of w
    R_1 : float
        Inner disc-radius sourcing material (au)
    R_2 : float
        Outer disc-radius sourcing material (au)

    Returns
    -------
    Jet mass loss rate in (M_sol / yr)
    """
    from .. import cnsts
    
    a = q_nd + q_nv

    # Avoid ZeroDivisionError
    if a == -1. or a == -2.:
        a *= 1. + 1e-12

    r2 = R_2 * con.au
    r1 = R_1 * con.au

    constant = 2. * con.pi * (mu * atomic_mass('H')) * (n_0 * 1e6) *\
               (v_0 * 1e3) * (w_0 * con.au) ** 2.

    return (constant *
            (r1 ** 2. + r2 * (r2 * (a + 1.) - r1 * (a + 2.)) * (r2 / r1) ** a) /
            ((r2 - r1) ** 2. * (a + 1.) * (a + 2.))) / cnsts.MSOL * con.year


def n_0_from_mlr(mlr: float, v_0: float, w_0: float, mu: float, q_nd: float,
                 q_nv: float, R_1: float, R_2: float) -> float:
    """
    Calculate central density at the base of the jet, given a mass loss rate

    Parameters
    ----------
    mlr : float
        Jet mass loss rate in (M_sol / yr)
    v_0 : float
        Central velocity at the base of the jet (km / s)
    w_0 : float
        Full jet-width at the base of the jet (au)
    mu : float
        Average atomic mass (m_H)
    q_nd : float
        Power-law exponent for number density as a function of w
    q_nv : float
        Power-law exponent for velocity as a function of w
    R_1 : float
        Inner disc-radius sourcing material (au)
    R_2 : float
        Outer disc-radius sourcing material (au)

    Returns
    -------
    Central number density at jet-base (cm^-3)
    """
    from .. import cnsts

    a = q_nd + q_nv

    # Avoid ZeroDivisionError
    if a == -1. or a == -2.:
        a *= 1. + 1e-12

    r2 = R_2 * con.au
    r1 = R_1 * con.au
    mlr_si = mlr * cnsts.MSOL / con.year

    constant = 2. * con.pi * (mu * atomic_mass('H')) * (v_0 * 1e3) *\
               (w_0 * con.au) ** 2.

    return mlr_si / constant /\
           ((r1 ** 2. + r2 * (r2 * (a + 1.) - r1 * (a + 2.)) * (r2 / r1) ** a) /
           ((r2 - r1) ** 2. * (a + 1.) * (a + 2.))) / 1e6


# ############################################################################ #
# ######################## Miscellaneous physics methods below ############### #
# ############################################################################ #
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
    :return: Rydberg constant (m^-1)
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


def nu_rrl(n: int, dn: int = 1, atom: str = "H") -> float:
    """
    Calculate radio recombination line frequency(s)

    Parameters
    ----------
    n
        Electronic transition number.

    dn
        Number of levels transitioned e.g. for alpha RRLs dn = 1

    atom
        Chemical symbol for atom to compute RRLs for e.g. 'H' for Hydrogen,
        'He' for Helium etc

    Returns
    -------
    nu0
        Rest frequency of the radio recombination line [Hz]

    """
    from .. import cnsts

    if atom not in cnsts.NZ:
        available = list(cnsts.NZ.keys())
        raise ValueError("RRL rest frequencies only available for "
                         f"{', '.join(available[:-1])} and {available[-1]}, "
                         f"not {atom}")

    n_p, n_n = cnsts.NZ[atom]

    mass = atomic_mass(atom)
    mass -= con.m_e * n_p
    r_m = con.Rydberg * (1. + con.m_e / mass) ** -1.

    nu0 = r_m * con.c * (1. / n ** 2. - 1. / (n + dn) ** 2.)

    return nu0


def vlsr_to_freq(vlsr: Union[float, npt.NDArray],
                 nu0: float) -> Union[float, npt.NDArray]:
    """
    Convert velocity (local standard of rest) to frequency

    Parameters
    ----------
    vlsr
        Velocity(s) [m/s]
    nu0
        Rest frequency [Hz]

    Returns
    -------
    nu
        Observed frequency(s) [Hz]
    """
    nu = nu0 * (1. - vlsr / con.c)

    return nu


def freq_to_vlsr(nu: Union[float, npt.NDArray],
                 nu0: float) -> Union[float, npt.NDArray]:
    """
    Convert frequency to velocity (local standard of rest)

    Parameters
    ----------
    nu
        Observed frequency(s) [Hz]
    nu0
        Rest frequency [Hz]

    Returns
    -------
    vlsr
        Velocity(s) [m/s]
    """
    vlsr = (1. - (nu / nu0)) * con.c

    return vlsr


def chanwidth_vlsr_to_hz(dvlsr: Union[float, npt.NDArray],
                         nu0: float) -> Union[float, npt.NDArray]:
    """
    Convert a channel width from m/s to Hz

    Parameters
    ----------
    dvlsr
        Channel width(s) [m/s]
    nu0
        Rest frequency [Hz]

    Returns
    -------
    dnu
        Channel width(s) [Hz]
    """
    dnu = nu0 - vlsr_to_freq(dvlsr, nu0)

    return dnu


def chanwidth_hz_to_vlsr(dnu: Union[float, npt.NDArray],
                         nu0: float) -> Union[float, npt.NDArray]:
    """
    Convert a channel width from Hz to m/s

    Parameters
    ----------
    dnu
        Channel width(s) [Hz]
    nu0
        Rest frequency [Hz]

    Returns
    -------
    dvlsr
        Channel width(s) [m/s]
    """
    return freq_to_vlsr(nu0 - dnu, nu0)


def atomic_mass(atom: str) -> float:
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
    from .. import cfg, cnsts

    ams = pd.read_pickle(cfg.dcys["files"] + os.sep + "atomic_masses.pkl")
    n_p, n_n = cnsts.NZ[atom]
    M = ams[(ams['N'] == n_n) & (ams['Z'] == n_p)]['mass[micro-u]'].values[0]
    M *= 1e-6 * con.u
    return M

def import_vanHoof2014(errors=False):
    from .. import cfg

    datafile = os.sep.join([cfg.dcys['files'], "vanHoofetal2014.data"])

    with open(datafile, 'rt') as f:
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

# ############################################################################ #
# ####### Shakura & Sunyaev (1973) accretion disc-related methods below ###### #
# ############################################################################ #
def u0_ss73(alpha, acc_rate, m_yso, radius, zone='c'):
    """
    Surface density of the disc according to Shakura & Sunyaev (1973)'s
    equation 2.19

    Parameters
    ----------
    alpha : float
        Alpha parameter of Shakura & Sunyaev (1973)
    acc_rate : float
        Accretion rate (solar masses per year)
    m_yso : float
        Central object mass (solar masses)
    radius : float
        Distance from central object (au)
    zone : str
        Which zone of the disc value for radius is located in

    Returns
    -------
    Disc surface density in g cm^-2
    """
    if zone != 'c':
        return ValueError("Only disc-zone c from Shakura & Sunyaev (1973) "
                          "is currently implemented")

    m_dot = acc_rate / 3e-8 * m_yso ** -1.
    r = m_yso ** -1. * (radius * con.au) / 9000.

    return (6.1e5 * alpha ** -0.8 * m_dot ** 0.7 * m_yso ** 0.2 *
            r ** -0.75 * (1. - r ** -0.5) ** 0.7)


def temp_ss73(alpha, acc_rate, m_yso, radius, zone='c'):
    """
    Temperature of the disc according to Shakura & Sunyaev (1973)'s equation
    2.19

    Parameters
    ----------
    alpha : float
        Alpha parameter of Shakura & Sunyaev (1973)
    acc_rate : float
        Accretion rate (solar masses per year)
    m_yso : float
        Central object mass (solar masses)
    radius : float
        Distance from central object (au)
    zone : str
        Which zone of the disc value for radius is located in

    Returns
    -------
    Temperature in Kelvin
    """
    if zone != 'c':
        return ValueError("Only disc-zone c from Shakura & Sunyaev (1973) "
                          "is currently implemented")

    m_dot = acc_rate / 3e-8 * m_yso ** -1.
    r = m_yso ** -1. * (radius * con.au) / 9000.

    return (8.6e7 * alpha ** -0.2 * m_dot ** 0.3 * m_yso ** -0.2 * r ** -0.75 *
            (1. - r ** -0.5) ** 0.3)


def z0_ss73(alpha, acc_rate, m_yso, radius, zone='c'):
    """
    Scale height of the disc according to Shakura & Sunyaev (1973)'s equation
    2.19

    Parameters
    ----------
    alpha : float
        Alpha parameter of Shakura & Sunyaev (1973)
    acc_rate : float
        Accretion rate (solar masses per year)
    m_yso : float
        Central object mass (solar masses)
    radius : float
        Distance from central object (au)
    zone : str
        Which zone of the disc value for radius is located in

    Returns
    -------
    Scale height in au
    """
    if zone != 'c':
        return ValueError("Only disc-zone c from Shakura & Sunyaev (1973) "
                          "is currently implemented")

    m_dot = acc_rate / 3e-8 * m_yso ** -1.
    r = m_yso ** -1. * (radius * con.au) / 9000.

    return (6.1e3 * alpha ** -0.1 * m_dot ** 0.15 * m_yso ** 0.9 *
            r ** (9. / 8.) * (1. - r ** -0.5) ** 0.15) / 1e2 / con.au


def n_ss73(alpha, acc_rate, m_yso, radius, zone='c'):
    """
    Number density of the disc according to Shakura & Sunyaev (1973)'s equation
    2.19

    Parameters
    ----------
    alpha : float
        Alpha parameter of Shakura & Sunyaev (1973)
    acc_rate : float
        Accretion rate (solar masses per year)
    m_yso : float
        Central object mass (solar masses)
    radius : float
        Distance from central object (au)
    zone : str
        Which zone of the disc value for radius is located in

    Returns
    -------
    Number density in cm ^ -3
    """
    if zone != 'c':
        return ValueError("Only disc-zone c from Shakura & Sunyaev (1973) "
                          "is currently implemented")

    m_dot = acc_rate / 3e-8 * m_yso ** -1.
    r = m_yso ** -1. * (radius * con.au) / 9000.

    return (3e25 * alpha ** -0.7 * m_dot ** 0.55 * m_yso ** -0.7 *
            r ** (-15. / 8.) * (1. - r ** -0.5) ** 0.55)


def tau_ss73(alpha, acc_rate, m_yso, radius, zone='c'):
    """
    Optical depth of the disc according to Shakura & Sunyaev (1973)'s
    equation 2.19

    Parameters
    ----------
    alpha : float
        Alpha parameter of Shakura & Sunyaev (1973)
    acc_rate : float
        Accretion rate (solar masses per year)
    m_yso : float
        Central object mass (solar masses)
    radius : float
        Distance from central object (au)
    zone : str
        Which zone of the disc value for radius is located in

    Returns
    -------
    Optical depth (dimensionless)
    """
    if zone != 'c':
        return ValueError("Only disc-zone c from Shakura & Sunyaev (1973) "
                          "is currently implemented")

    m_dot = acc_rate / 3e-8 * m_yso ** -1.
    r = m_yso ** -1. * (radius * con.au) / 9000.

    return (3.4e2 * alpha ** -0.8 * m_dot ** 0.2 * m_yso ** 0.2 *
            (1. - r ** -0.5) ** 0.2)


def vr_ss73(alpha, acc_rate, m_yso, radius, zone='c'):
    """
    Radial velocity in the disc according to Shakura & Sunyaev (1973)'s equation
    2.19

    Parameters
    ----------
    alpha : float
        Alpha parameter of Shakura & Sunyaev (1973)
    acc_rate : float
        Accretion rate (solar masses per year)
    m_yso : float
        Central object mass (solar masses)
    radius : float
        Distance from central object (au)
    zone : str
        Which zone of the disc value for radius is located in

    Returns
    -------
    Radial velocity in cm / s
    """
    if zone != 'c':
        return ValueError("Only disc-zone c from Shakura & Sunyaev (1973) "
                          "is currently implemented")

    m_dot = acc_rate / 3e-8 * m_yso ** -1.
    r = m_yso ** -1. * (radius * con.au) / 9000.

    return (5.8e5 * alpha ** 0.8 * m_dot ** 0.3 * m_yso ** -0.2 *
            r ** -0.25 * (1. - r ** -0.5) ** -0.7)


def h_ss73(alpha, acc_rate, m_yso, radius, zone='c'):
    """
    Magnetic-field upper-limit in the disc according to Shakura & Sunyaev
    (1973)'s equation 2.19

    Parameters
    ----------
    alpha : float
        Alpha parameter of Shakura & Sunyaev (1973)
    acc_rate : float
        Accretion rate (solar masses per year)
    m_yso : float
        Central object mass (solar masses)
    radius : float
        Distance from central object (au)
    zone : str
        Which zone of the disc value for radius is located in

    Returns
    -------
    Magnetic field upper-limit in Gauss
    """
    if zone != 'c':
        return ValueError("Only disc-zone c from Shakura & Sunyaev (1973) "
                          "is currently implemented")

    m_dot = acc_rate / 3e-8 * m_yso ** -1.
    r = m_yso ** -1. * (radius * con.au) / 9000.

    return (2.1e9 * alpha ** 0.05 * m_dot ** 0.425 * m_yso ** -0.45 *
            r ** (-21. / 16.) * (1. - r ** -0.5) ** 0.425)
