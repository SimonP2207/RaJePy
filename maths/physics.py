# -*- coding: utf-8 -*-
import os
import numpy as np
import scipy.constants as con
import pandas as pd
from mpmath import gammainc

from RaJePy import _config as cfg
from RaJePy import _constants as cnsts

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

    R_M = con.Rydberg * (1. + con.m_e / M)**-1.
    return R_M * con.c * (1. / n**2. - 1. / (n + dn)**2.)

def fwhm_rrl(n, dn=1, atom="H", T_e=1e4):
    """
    Full-width at half-maximum for radio recombination line(s)

    Parameters
    ----------
    nu_0 : float or Iterable
        Frequency of radio recombination line.
    T_e : float
        Electron temperature. The default is 1e4.

    Returns
    -------
    FWHM of RRL in Hz

    """
    M = atomic_mass(atom)
    
    delta_nu = np.sqrt(8. * np.log(2.) * con.k / con.c**2.)
    delta_nu *= np.sqrt(T_e / M) * nu_rrl(n, dn, atom)
    
    return delta_nu

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
    a_j, a_k = 6.5E-38, 0.212  # given as constants of cgs equations
    c = (1. + jm.params['geometry']['epsilon'] +
         jm.params['power_laws']['q_T']) / jm.params['power_laws']['q_tau']
    flux = 2**(1. - c) * (jm.params['target']['dist'] * con.parsec * 1e2)**-2.
    flux *= a_j * a_k**(-1. - c) * jm.params['properties']['T_0']**(1. + 1.35 * c)
    flux *= jm.params['geometry']['r_0'] * con.au * 1e2
    flux *= (jm.params['geometry']['w_0'] * con.au * 1e2)**(1. - c)
    flux *= (jm.params['properties']['n_0'] *
             jm.params['properties']['x_0'])**(-(2. * c))
    flux *= np.sin(np.radians(jm.params['geometry']['inc']))**(1. + c) / \
            (c * (1. + jm.params['geometry']['epsilon'] +
            jm.params['power_laws']['q_T'] +
            jm.params['power_laws']['q_tau']))
    alpha = 2. + (2.1 / jm.params['power_laws']['q_tau']) * \
            (1 + jm.params['geometry']['epsilon'] +
            jm.params['power_laws']['q_T'])

    flux *= freq**alpha  # in erg cm^-2 s^-1 Hz^-1
    flux *= 1e-7 * 1e2**2.  # now in W m^-2 Hz^-1
    return flux / 1e-26  # now in Jy

def flux_expected_r86(jm, freq, y_max):
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

    Returns
    -------
    float
        Exact flux expected from Reynolds (1986)'s analytical model (Jy).

    """
    a_j, a_k = 6.5E-38, 0.212  # given as constants of cgs equations
    d = jm.params['target']['dist'] * con.parsec * 1e2  # cm
    y_max = np.tan(y_max * con.arcsec) * d  # in cm
    inc = jm.params['geometry']['inc']  # degrees
    r_0 = jm.params['geometry']['r_0'] * con.au * 1e2  # cm
    y_0 = r_0 * np.sin(np.radians(inc))  # cm
    w_0 = jm.params['geometry']['w_0'] * con.au * 1e2  # cm
    d = jm.params['target']['dist'] * con.parsec * 1e2  # cm
    T_0 = jm.params['properties']['T_0']  # K
    n_0 = jm.params['properties']['n_0']  # cm^-3
    chi_0 = jm.params['properties']['x_0']  # dimensionless
    q_tau = jm.params["power_laws"]["q_tau"]  # dimensionless
    q_T = jm.params["power_laws"]["q_T"]  # dimensionless
    eps = jm.params["geometry"]["epsilon"]  # dimensionless
    
    tau_0 = 2. * a_k * w_0 * n_0**2. * chi_0**2. * T_0**-1.35 * freq**-2.1 * \
            np.sin(np.radians(inc))**-1.
    
    c = 1. + eps + q_T
    
    def indef_integral(yval):
        tau = tau_0 * (yval / y_0)**q_tau
        val = y_0**-(eps + q_T) * yval**c
        val *= (c * tau**(-c / q_tau) * gammainc(c / q_tau, tau)) + q_tau
        val /= q_tau * c
        return float(val)
    
    flux = 2. * w_0 * d**-2. * a_j * a_k**-1. * T_0 * freq**2.
    flux *= indef_integral(y_max) - indef_integral(y_0)  # erg cm^-2 s^-1 Hz^-1
    flux *= 1e-7 * 1e2**2.  # W m^-2 Hz^-1
    return flux / 1e-26  # Jy

def import_vanHoof2014(errors=False):
    if errors:
        from  uncertainties import ufloat as uf

    datafile = os.sep.join([os.path.expanduser('~'), "Dropbox",
                            "SpyderProjects", "vajepy", "files",
                            "vanHoofetal2014.data"]) 

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
                            np.round(logu_start + (step * (n_logu - 1)), decimals=1),
                            n_logu)

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
    Ry = con.m_e * con.e**4. / (8 * con.epsilon_0**2. * con.h**2.)

    logg2 = np.log10(z**2. * Ry / (con.k * temp))
    logu = np.log10(con.h * freq / (con.k * temp))

    from scipy.interpolate import interp2d
    logg2s, logus, gffs = import_vanHoof2014(errors=False)
    
    col = np.argmin(np.abs(logg2s[0] - logg2))
    row = np.argmin(np.abs(logus[:,0] - logu))
    
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


def tau_r(jm, freq, r):
    """
    Optical depth from Equations 4 + 5 of Reynolds (1986) analytical model
    paper, for monopolar jet.

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
        Exact flux expected from Reynolds (1986)'s analytical model (Jy).

    """
    a_j, a_k = 6.5E-38, 0.212  # given as constants of cgs equations
    d = jm.params['target']['dist'] * con.parsec * 1e2  # cm
    inc = jm.params['geometry']['inc']  # degrees
    r *= con.au * 1e2  # cm
    r_0 = jm.params['geometry']['r_0'] * con.au * 1e2  # cm
    y_0 = r_0 * np.sin(np.radians(inc))  # cm
    w_0 = jm.params['geometry']['w_0'] * con.au * 1e2  # cm
    d = jm.params['target']['dist'] * con.parsec * 1e2  # cm
    T_0 = jm.params['properties']['T_0']  # K
    n_0 = jm.params['properties']['n_0']  # cm^-3
    chi_0 = jm.params['properties']['x_0']  # dimensionless
    q_tau = jm.params["power_laws"]["q_tau"]  # dimensionless

    tau_0 = 2. * a_k * w_0 * n_0 ** 2. * chi_0 ** 2. * T_0 ** -1.35 * \
            freq** -2.1 * np.sin(np.radians(inc)) ** -1.

    return tau_0 * (r / r_0)**q_tau
