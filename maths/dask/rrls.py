from typing import Callable, Tuple, Iterable, Union
import numpy as np
import scipy.constants as con
from scipy.special import erf, wofz
import dask.array as da
from . import physics as phys
from ..rrls import rrl_nu_0, energy_n, f_n1n2, ni_from_ne, deltanu_l, deltanu_v, phi_thermal_nu_integrated, phi_stark_nu_integrated, phi_voigt_nu_integrated

c_cgs = con.c * 1e2
h_cgs = con.h * 1e7
e_cgs = con.e * (con.c * 10.)
m_e_cgs = con.m_e * 1e3
k_cgs = con.k * 1e7

ni_from_ne.__annotations__['n_e'] = da.core.Array
deltanu_l.__annotations__['n_e'] = da.core.Array


def deltanu_g(nu_0: float, temp: Union[float, da.core.Array],
              atom: str) -> Union[float, da.core.Array]:
    """
    Full-width at half maximum line width (Hz) for purely thermal (Doppler),
    Gaussian broadening. Equation 2.21 of Gordon & Sorochenko.

    :param float nu_0: RRL Doppler-shifted rest frequency (Hz)
    :param float temp: Electron temperature (K)
    :param str atom: Atomic nucleus' element as per periodic table symbol
    :return: Delta nu_G (Hz)
    :rtype: float
    """
    m = phys.atomic_mass(atom)
    return da.sqrt(
        4. * np.log(2.) * 2. * con.k * temp / (m * con.c ** 2.)) * nu_0

# TODO: Still to parallelise the below functions
def phi_voigt_nu_integrated(freq: float, nu_0: float, fwhm_thermal: float,
                            fwhm_stark: float, bw: float, n_sum: int = 30,
                            tau_m: float = 12., average: bool = True) -> float:
    """
    Computation of the indefinite integral of the Voigt profile integrated
    in x, I(x,y) = integral K(x,y) dx as per Quine & Abrarov (2013)

    :param float freq: Frequency of observation (Hz)
    :param float nu_0: Doppler-corrected rest frequency (Hz)

    :param float fwhm_thermal: Full-width at half maximum line width (Hz) for
                               purely thermal (Doppler) broadenening
    :param float fwhm_stark: Full-width at half-maximum of the line (Hz) for
                             purely collisional (Stark) broadenening
    :param float bw: Bandwidth of the observation (Hz)
    :param int n_sum: Number of summations (default = 30)
    :param float tau_m: Margin value, tau_m (default = 12.0)
    :param bool average: Return average value of Voigt profile over the
                         bandwidth? If False, returns area under curve
                         instead. (default = True)
    :return: I(x,y)
    :rtype: float
    """
    y = _y_qa13(fwhm_stark, fwhm_thermal)
    ns = np.arange(1, n_sum + 1)
    pm = np.pi / tau_m

    def indef_integral(nu):
        x = _x_qa13(nu, nu_0, fwhm_thermal)

        # Central region (see Figure 7 of Quine & Abrarov, 2013)
        in_central_region = np.abs(x + y * 4.) <= 4.5

        # External region (see Figure 7 of Quine & Abrarov, 2013)
        in_external_region = np.abs(x + y / 1.54545) > 11

        if not in_central_region:
            if not in_external_region:
                # Equation 15 of Quine & Abrarov (2013)
                p1 = 1. / tau_m * np.arctan(x / y)

                # Summation
                p2 = np.sum((np.arctan((x + ns * pm) / y) +
                             np.arctan((x - ns * pm) / y)) *
                            _an_qa13(ns, tau_m)) * 0.28209479177387814

                result = p1 + p2
            else:
                # Equation 17 of Quine & Abrarov (2013)
                p1 = np.arctan(x / y) / 1.7724538509
                p2 = x * y / (2. * 1.7724538509 * (x ** 2. + y ** 2.) ** 2.)

                result = p1 - p2

        else:
            # Equation 11 of Quine & Abrarov (2013)
            exp_negtauy = np.exp(-tau_m * y)
            tauy2 = tau_m ** 2. * y ** 2.

            p1 = 0.8862269254527579
            p2a = _bn_qa13(x, 0, tau_m) * (1. - exp_negtauy) / (2. * tau_m * y)

            # p2b = 0.
            # for n in range(1, n_sum + 1):  # Summation
            #     val = _bn_qa13(x, n, tau_m) * (-1. ** n * exp_negtauy - 1.)
            #     p2b += val / (n ** 2. * np.pi ** 2. + tauy2)
            # p2b *= tau_m * y

            p2b = np.sum((_bn_qa13(x, ns, tau_m) *
                          (-1. ** ns * exp_negtauy - 1.)) /
                         (ns ** 2. * np.pi ** 2. + tauy2)) * tau_m * y

            result = p1 * (p2a - p2b)

        # Factor of sqrt(pi) here for difference between Voigt function and
        # Voigt profile (private communication with S. M. Abrarov)
        return result / 1.7724538509055159

    eval_int = indef_integral(freq + bw / 2.) - indef_integral(freq - bw / 2.)

    return eval_int / (bw if average else 1.)
phi_voigt_nu_integrated = np.vectorize(phi_voigt_nu_integrated)


def phi_thermal_nu(nu_0: float, fwhm_thermal: float,
                   freq: Union[Iterable, float, None] = None) \
        -> Union[Callable, Iterable, float]:
    """
    Normalised line profile for purely Doppler-broadened line. Equation 2.20
    of Gordon & Sorochenko.\

    :param float nu_0: RRL Doppler-shifted rest frequency (Hz)
    :param float fwhm_thermal: Full-width at half maximum line width (Hz) for
                               purely thermal (Doppler) broadenening
    :param float freq: Frequency of observation (Hz). Default is None, in which
                       case the function, phi_V (nu) is returned
    :return: phi_G (nu) if arg freq is None else phi_G (Hz^-1)
    :rtype: function, Iterable or float
    """

    def func(nu):
        p1 = np.sqrt(4. * np.log(2.) / np.pi) / fwhm_thermal
        p2 = np.exp(-4. * np.log(2.) * ((nu_0 - nu) / fwhm_thermal) ** 2.)
        return p1 * p2

    if freq is None:
        return func

    return func(freq)


def phi_stark_nu(nu_0: float, fwhm_stark: float,
                 freq: Union[Iterable, float, None] = None) \
        -> Union[Callable, Iterable, float]:
    """
    Normalised, Lorentzian line profile for purely Stark-broadened line
    integrated spectrally. From equation 2.30 of Gordon & Sorochenko.

    :param float nu_0: Doppler-corrected rest frequency (Hz)
    :param float fwhm_stark: Full-width at half-maximum of the line (Hz) for
                             purely collisional (Stark) broadenening
    :param float freq: Frequency of observation (Hz). Default is None, in which
                       case the function, phi_V (nu) is returned
    :return: phi_L (nu) if arg freq is None else phi_L (Hz^-1)
    :rtype: function, Iterable or float
    """

    def func(nu):
        num = 2. * fwhm_stark
        den = np.pi * (4. * (nu - nu_0) ** 2. + fwhm_stark ** 2.)
        return num / den

    if freq is None:
        return func

    return func(freq)


def phi_voigt_nu(nu_0: float, fwhm_stark: Union[Iterable, float],
                 fwhm_thermal: Union[Iterable, float],
                 freq: Union[Iterable, float, None] = None) \
        -> Union[Callable, Iterable, float]:
    """
    Normalised, Voigt line profile for contributions of thermal and Stark
    broadening. This is the convolution of the Gaussian and Lorentzian line
    profiles. Taken from
    https://scipython.com/book/chapter-8-scipy/examples/the-voigt-profile/

    :param float nu_0: Doppler-corrected rest frequency (Hz)
    :param float fwhm_stark: Full-width at half-maximum of the line (Hz) for
                             purely collisional (Stark) broadenening
    :param float fwhm_thermal: Full-width at half maximum line width (Hz) for
                               purely thermal (Doppler) broadenening
    :param float freq: Frequency of observation (Hz). Default is None, in which
                       case the function, phi_V (nu) is returned
    :return: phi_V (nu) if arg freq is None else phi_V (Hz^-1)
    :rtype: function, Iterable or float
    """

    def func(nu):
        sigma = fwhm_thermal / 2. / np.sqrt(2. * np.log(2))

        return np.real(wofz(((nu - nu_0) + 1j * fwhm_stark / 2.) /
                            sigma / np.sqrt(2.))) / sigma / np.sqrt(2. * np.pi)

    if freq is None:
        return func

    return func(freq)


def kappa_l(freq: float, n: int, oscillator_strength: float,
            line_profile_contribution: float, n_e: float, n_i: float,
            temp: Union[float, np.ndarray],
            z: int, energy_n1: float) -> Union[float, np.ndarray]:
    """
    RRL absorption coefficient, kappa_L (cm^-1). Equation 2.114 of Gordon &
    Sorochenko. Note that all calculations are performed in cgs units.

    :param float freq: Frequency of observation (Hz)
    :param int n: RRL transition lower electronic level
    :param float oscillator_strength: Absorption oscillator strength
    :param float line_profile_contribution: Absorption of the line over range
                                            in bandwidth (Hz^-1)
    :param float n_e: Electron number density (cm^-3)
    :param float n_i: Ion number density (cm^-3)
    :param float temp: Electron temperature (K)
    :param int z: Ion atomic number
    :param float energy_n1: Energy of the RRL lower electronic level (erg)
    :return: kappa_L (cm^-1)
    :rtype: float
    """
    p0 = 1.0991132675738456e-17
    p1 = n ** 2. * oscillator_strength * line_profile_contribution
    p2 = n_e * n_i / temp ** 1.5
    p3 = np.exp((z ** 2. * energy_n1) / (k_cgs * temp))
    p4 = 1. - np.exp(-h_cgs * freq / (k_cgs * temp))

    return p0 * p1 * p2 * p3 * p4


def kappa_l_average(freq: float, n: int, oscillator_strength: float,
                    av_line_profile_contribution: float, n_e: float, n_i: float,
                    temp: float, z: int, energy_n1: float, bw: float) -> float:
    """
    RRL absorption coefficient, kappa_L (cm^-1). Equation 2.114 of Gordon &
    Sorochenko. Note that all calculations are performed in cgs units.

    :param float freq: Frequency of observation (Hz)
    :param int n: RRL transition lower electronic level
    :param float oscillator_strength: Absorption oscillator strength
    :param float av_line_profile_contribution: Average distribution of the line
                                               over bandwidth (Hz^-1)
    :param float n_e: Electron number density (cm^-3)
    :param float n_i: Ion number density (cm^-3)
    :param float temp: Electron temperature (K)
    :param int z: Ion atomic number
    :param float energy_n1: Energy of the RRL lower electronic level (erg)
    :param float bw: Bandwidth of the observation (Hz)
    :return: kappa_L (cm^-1)
    :rtype: float
    """
    p0 = 1.0991132675738456e-17
    p1 = n ** 2. * oscillator_strength * av_line_profile_contribution
    p2 = n_e * n_i / temp ** 1.5
    p3 = np.exp((z ** 2. * energy_n1) / (k_cgs * temp))

    def indef_integral(nu):
        return  k_cgs * temp * np.exp(-h_cgs * nu /
                                      (k_cgs * temp)) / h_cgs + nu

    eval_int = indef_integral(freq + bw / 2.) - indef_integral(freq - bw / 2.)

    return p0 * p1 * p2 * p3 * eval_int / bw
kappa_l_average = np.vectorize(kappa_l_average)


def line_intensity_lte(freq: Union[float, Iterable, np.ndarray],
                       temp: Union[float, Iterable, np.ndarray],
                       tau_c: Union[float, Iterable, np.ndarray],
                       tau_l: Union[float, Iterable, np.ndarray]) -> Union[float, Iterable, np.ndarray]:
    """
    Intensity of the RRL assuming local thermodynamic equilibrium (LTE).
    Equation 2.122 of Gordon & Sorochenko (2002).

    :param float freq: Observing frequency (Hz)
    :param float temp: Electron temperature (K)
    :param float tau_c: Optical depth for continuum emission
    :param float tau_l: Optical depth of RRL in LTE
    :return: I_L (W m^-2 Hz^-1 sr^-1)
    :rtype: float

    """
    b_nu = phys.blackbody_nu(freq, temp)  # erg s^-1 cm^-2 Hz^-1 sr^-1
    i_l_cgs = b_nu * np.exp(-tau_c) * (1. - np.exp(-tau_l))

    i_l = i_l_cgs * 1e-7 * 1e4  # cgs to S.I.

    return i_l


def line_continuum_ratio_lte(tau_c: float, tau_l: float) -> float:
    """
    RRL Line to continuum ratio assuming local thermodynamic equilibrium (LTE).
    From equations 2.121 and 2.122 of Gordon & Sorochenko
    (2002).

    :param float tau_c: Optical depth for continuum emission
    :param float tau_l: Optical depth of RRL in LTE
    :return: I_L / I_C (dimensionless)
    :rtype: float

    """
    return (1. - np.exp(-tau_l)) / (np.exp(tau_c) - 1.)


def beta_coeff(freq: float, temp: float, b_n1: float, b_n2: float) -> float:
    """
    Beta coefficient for non-LTE RRL intensity calculations. Equation 2.130
    from Gordon & Sorochenko (2002).

    :param float freq: Observing frequency (Hz)
    :param float temp: Electron temperature (K)
    :param float b_n1: Departure coefficient for lower electronic level of
                       transition
    :param float b_n2: Departure coefficient for upper electronic level of
                       transition
    :return: :math:`\beta`
    :rtype: float
    """
    exp = np.exp(-con.h * freq / (con.k * temp))
    num = 1. - (b_n2 / b_n1) * exp
    den = 1. - exp

    return num / den


def tau_nu_coeff(beta: float, tau_c: float, tau_l_star: float,
                 b_n1: float) -> float:
    """
    Combined continuum and RRL non-LTE optical depth. Equation 2.141 from
    Gordon & Sorochenko (2002).

    :param float beta: Beta coefficient for non-LTE RRL intensity calculations
    :param float tau_c: Optical depth for continuum emission
    :param float tau_l_star: Optical depth of RRL in LTE
    :param float b_n1: Departure coefficient for lower electronic level of
                       transition
    :return: :math:`\tau_\nu`
    :rtype: float
    """
    return tau_c + tau_l_star * b_n1 * beta


def eta_coeff(b_n1: float, b_n2: float, kappa_c: float, kappa_l_star: float,
              beta: float) -> float:
    """
    Eta coefficient for non-LTE RRL intensity calculations. Equation 2.139
    from Gordon & Sorochenko (2002).

    :param float b_n1: Departure coefficient for lower electronic level of
                       transition
    :param float b_n2: Departure coefficient for upper electronic level of
                       transition
    :param float kappa_c: Absorption coefficient for continuum emission (cm^-1)
    :param float kappa_l_star: Absorption coefficient of RRL in LTE  (cm^-1)
    :param float beta: Beta coefficient for non-LTE RRL intensity calculations
    :return: eta
    :rtype: float
    """
    kappa_ratio = kappa_l_star / kappa_c
    num = 1. + b_n2 * kappa_ratio
    den = 1. + b_n1 * kappa_ratio * beta

    return num / den


def line_continuum_ratio_nonlte(eta: float, tau_nu: float,
                                tau_c: float) -> float:
    """
    RRL intensity to continuum intensity ratio (non-LTE). Equation 2.140 from
    Gordon & Sorochenko (2002).

    :param float eta: Eta coefficient for non-LTE RRL intensity calculations
    :param float tau_nu: Combined continuum and RRL non-LTE optical depth
    :param float tau_c: Continuum RRL optical depth
    :return: I_L / I_C
    :rtype: float
    """
    num = eta * (1. - np.exp(-tau_nu))
    den = 1. - np.exp(-tau_c)

    return num / den - 1.


def _x_qa13(freq: float, nu_0: float, fwhm_thermal: float) -> float:
    """
    Equation for x-variable. See section 2.1 of Quine & Abrarov (2013).

    :param float freq: Observing frequency (Hz)
    :param float nu_0: Doppler-corrected rest frequency (Hz)
    :param float fwhm_thermal: Full-width at half maximum line width (Hz) for
                               purely thermal (Doppler) broadenening
    :returns: x (dimensionless)
    :rtype: float
    """
    # Extra factor of two for conversion of fwhm to hwhm
    return 1.6651092223153954 * (freq - nu_0) / fwhm_thermal


def _y_qa13(fwhm_stark: float, fwhm_thermal: float) -> float:
    """
    Equation for y-variable. See section 2.1 of Quine & Abrarov (2013).

    :param float fwhm_stark: Full-width at half-maximum of the line (Hz) for
                             purely collisional (Stark) broadenening
    :param float fwhm_thermal: Full-width at half maximum line width (Hz) for
                               purely thermal (Doppler) broadenening
    :returns: y (dimensionless)
    :rtype: float
    """
    return 0.8325546111576977 * fwhm_stark / fwhm_thermal


def _an_qa13(n: int, tau_m: float) -> float:
    """
    Fourier expansion coefficients, a_n for use in computation of
    indefinite integral of the Voigt function. Equation 12 of Quine & Abrarov
    (2013).

    :param int n: n-parameter from summation
    :param float tau_m: Margin value, tau_m
    :return: a_n (x)
    :rtype: float
    """
    return 3.5449077018110318 / tau_m * np.exp(-(n * np.pi / tau_m) ** 2.)


def _bn_qa13(x: float, n: int, tau_m: float) -> float:
    """
    Fourier expansion coefficients, b_n, for use in computation of
    indefinite integral of the Voigt function. Equation 12 of Quine & Abrarov
    (2013).

    :param float x: x-parameter
    :param int n: n-parameter from summation
    :param float tau_m: Margin value, tau_m
    :return: b_n (x)
    :rtype: float
    """
    npt = n * np.pi / tau_m
    return erf(npt + x) - erf(npt - x)


def rrl_parser(rrl_str: str) -> Tuple[str, int, int]:
    """
    Parser an RRL string into the relevant element, lower transition level
    and delta n e.g. 'H58a' will return ('H', 58, 1)

    :param str rrl_str: Notation for RRL e.g. H58a, He42b etc.
    :return: Element, n-number, dn
    :rtype: tuple
    """
    rrl_parser_element = ''
    rrl_parser_n = ''
    rrl_parser_dn = {'a': 1, 'b': 2, 'g': 3, 'd': 4}[rrl_str[-1].lower()]

    for char in rrl_str[:-1]:
        if char.isalpha():
            rrl_parser_element += char
        else:
            rrl_parser_n += char

    return rrl_parser_element, int(rrl_parser_n), rrl_parser_dn


if __name__ == '__main__':
    # Example usage below
    # import matplotlib.pylab as plt

    # ######################################################################## #
    # ############################## Input ################################### #
    # ######################################################################## #
    line = 'H98a'
    shape = (10, 10, 10)
    t_e = np.full(shape, 1e4)  # K
    v_lsr = np.full(shape, 100.)  # km/s
    num_dens_e = np.full(shape, 1e7)  # cm^-3
    chan_width = 1e6  # Hz
    bandwidth = .1e9  # Hz
    cell_size = 0.5  # au
    obs_freq = rrl_nu_0(*rrl_parser(line))
    obs_freq = phys.doppler_shift(obs_freq, 100.)
    # ######################################################################## #
    # ############################## Logic ################################### #
    # ######################################################################## #
    # General
    element, rrl_n, rrl_dn = rrl_parser(line)
    rest_freq = phys.doppler_shift(rrl_nu_0(element, rrl_n, rrl_dn), v_lsr)

    # freq_range = rest_freq + np.array([-0.5, 0.5]) * bandwidth
    # nus = np.linspace(*freq_range, 1000)

    # Continuum
    gff = 11.95 * t_e ** 0.15 * obs_freq ** -0.1

    # Equation 1.26 and 5.19b of Rybicki and Lightman (cgs)
    kappa_ff = 0.018 * t_e ** -1.5 * phys.z_number(element) * num_dens_e *\
               ni_from_ne(num_dens_e) * obs_freq ** -2. * gff
    tau_ff = kappa_ff * (cell_size * con.au * 1e2)

    tb_ff = t_e * (1. - np.exp(-tau_ff))
    i_ff = 2. * obs_freq ** 2. * con.k * tb_ff / con.c ** 2.

    # RRL
    rrl_fwhm_thermal = deltanu_g(rest_freq, t_e, element)
    fn1n2 = f_n1n2(rrl_n, rrl_dn)
    en = energy_n(rrl_n, element)
    z_atom = phys.z_number(element)
    bnut = phys.blackbody_nu(obs_freq, t_e)
    rrl_fwhm_stark = deltanu_l(num_dens_e, rrl_n, rrl_dn)

    # 'Exact' solutions
    # k_xy_integrated = phi_voigt_nu_integrated(obs_freq, rest_freq,
    #                                           rrl_fwhm_thermal, rrl_fwhm_stark,
    #                                           chan_width)
    #
    #
    #
    # kappa_rrl_lte_av = kappa_l_average(obs_freq, rrl_n, fn1n2,
    #                                    k_xy_integrated, num_dens_e,
    #                                    ni_from_ne(num_dens_e, element), t_e,
    #                                    z_atom, en, chan_width)

    # Approximate solutions
    phi_v = phi_voigt_nu(rest_freq, rrl_fwhm_stark, rrl_fwhm_thermal)
    kappa_rrl_lte = kappa_l(obs_freq, rrl_n, fn1n2, phi_v(obs_freq),
                            num_dens_e, ni_from_ne(num_dens_e, element), t_e,
                            z_atom, en)

    tau_rrl_lte = kappa_rrl_lte * cell_size * con.au * 1e2

    # LTE assumption
    i_rrl_lte = line_intensity_lte(obs_freq, t_e, tau_ff, tau_rrl_lte)

    # Non-LTE

    # Equation for line to continuum ratio for consistency checks
    fwhm_voigt = deltanu_v(rrl_fwhm_thermal, rrl_fwhm_stark)
    fwhm_voigt_mps = fwhm_voigt * con.c / rest_freq
    yplus = 0.1  # Helium to hydrogen ratio
    lc_ratio = (.28 * (rest_freq / 1e9) ** 1.1 * (t_e / 1e4) ** -1.1 *
                (fwhm_voigt_mps / 1e3) ** -1. * (1. + yplus) ** -1. + 1.)
    lc_ratio = lc_ratio ** (2. / 3.) - 1.
    # ######################################################################## #
    # ############################## Plotting ################################ #
    # ######################################################################## #
    # plt.close('all')
    #
    # fig, ax = plt.subplots(1, 1, figsize=(5., 5.))
    #
    # ax.plot(nus, phi_v(nus), ls='-', lw=2, color='cornflowerblue', zorder=2)
    # ylims = [0, ax.get_ylim()[1]]
    # ax.vlines(obs_freq + np.array([-chan_width, chan_width]) / 2.,
    #           [ylims[0]] * 2, [ylims[1]] * 2, ls='-', color='firebrick',
    #           lw=0.5, zorder=3)
    # ax.set_xlim(freq_range)
    # ax.set_ylim(ylims)
    #
    # ax.set_xlabel(r'$\nu\,\left[\mathrm{Hz}\right]$')
    # ax.set_ylabel(r'$\phi\left(\nu\right)'
    #               r'\,\left[\mathrm{Hz}^{-1}\right]$')
    #
    # ax.set_title(r'$\mathrm{{{}}}{}{}$'.format(element, rrl_n,
    #                                            {1: r'\alpha',
    #                                             2: r'\beta',
    #                                             3: r'\gamma',
    #                                             4: r'\delta'}[rrl_dn]))
    #
    # plt.show()
