#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 10:32:29 2020

@author: simon
"""
import os
import numpy as np
import scipy.constants as con
import matplotlib.pylab as plt
from mpmath import gammainc
from RaJePy import JetModel


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

def flux_expected_r86_alt(jm, freq, y_max):
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

    def w_y(y):
        return w_0 * (y / y_0)**eps

    def T_y(y):
        return T_0 * (y / y_0)**q_T
    
    def tau_y(y):
        return tau_0 * (y / y_0)**q_tau
    
    def func(y):
        part1 = 2. * w_y(y) / d**2.
        part2 = a_j / a_k * T_y(y) * freq**2.
        part3 = 1. - np.exp(-tau_y(y))
        return part1 * part2 * part3
    func = np.vectorize(func)

    import scipy.integrate as integrate

    log_xs = np.logspace(np.log10(y_0), np.log10(y_max), 100)
    flux = integrate.cumtrapz(func(log_xs), log_xs)
    flux *= 1e-7 * 1e2**2.  # W m^-2 Hz^-1
    return flux / 1e-26  # Jy
    

jet_param_file = os.sep.join([os.path.expanduser("~"),
                              "Dropbox", "SpyderProjects",
                              "VaJePy", "files",
                              "example-model-params.py"])

jm = JetModel(jet_param_file)

plt.close('all')

freqs = np.logspace(8, 13, 1000)

fluxes1 = [flux_expected_r86(jm, f, 1.) for f in freqs]
fluxes2 = [flux_expected_r86_alt(jm, f, 1.)[-1] for f in freqs]

plt.loglog(freqs, fluxes1, 'b-')
plt.loglog(freqs, fluxes2, 'r--')

plt.show()
