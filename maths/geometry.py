# -*- coding: utf-8 -*-
"""
Module handling all mathematical functions and methods
"""

import numpy as np
from collections.abc import Iterable
import scipy.constants as con
from scipy.special import hyp2f1


def mod_r_0(opang, epsilon, w_0):
    """
    Calculates 'modified' launching radius i.e. the radius, r_0, at which a
    standard Reynolds (1986) jet would have a width, w_0, whilst maintaining a
    given opening angle

    Parameters
    ----------
    opang : float
        Opening angle of jet (deg)
    epsilon : float
        Power-law exponent for jet-width with distance along jet
    w_0 : float
        Half-width of jet-base (any physical unit)

    Returns
    -------
    'Modified' launching radius in same physical units as given for arg w_0
    """
    return epsilon * w_0 / np.tan(np.radians(opang) / 2.)

def t_rw(r, w, params):
    """
    Function to return time as a function of position (r, w) in a jet

    Parameters
    ----------
    r: float
        Jet radius coordinate in au
    w: float
        Jet width coordinate in au
    params: dict
        Parameter dictionary containing all relevant jet physical parameters

    Returns
    -------
    Time in years
    """
    w_0 = params['geometry']['w_0'] * con.au
    r_0 = params['geometry']['r_0'] * con.au
    v_0 = params["properties"]["v_0"] * 1e3
    mod_r_0 = params['geometry']['mod_r_0'] * con.au
    eps = params['geometry']['epsilon']
    R_1 = params["target"]["R_1"] * con.au
    R_2 = params["target"]["R_2"] * con.au
    q_v = params["power_laws"]["q_v"]
    q_vd = params["power_laws"]["q^d_v"]

    # If w == 0 and q_vd < 0, NaN is ultimately returned, therefore change
    # q_vd to 0 when w == 0 since it doesn't make any difference
    if w == 0:
        q_vd = 0.

    # Assume biconical symmetry in ejection velocities as negative values for
    # r results in return of NaNs
    if r < 0:
        r = np.abs(r)

    def indef_intgrl(r, w):
        """
        Indefinite integral in r
        """
        const = (w * con.au) * mod_r_0 ** eps * (R_2 - R_1) / (R_1 * w_0)
        rad = (r * con.au) + mod_r_0 - r_0
        num1 = rad ** (1. - q_v)
        num2 = (1. + const * rad ** (-eps)) ** (-q_vd)
        num3 = ((const + rad ** eps) / const) ** q_vd
        num4 = hyp2f1(q_vd, (1. + eps * q_vd - q_v) / eps,
                      (1. + eps + eps * q_vd - q_v) / eps,
                      -(rad ** eps / const))
        den = 1. + eps * q_vd - q_v
        return mod_r_0 ** q_v / v_0 * num1 * num2 * num3 * num4 / den

    return (indef_intgrl(r, w) - indef_intgrl(r_0 / con.au, w)) / con.year

def disc_angle(x, y, z,
               inc, pa):
    """Returns position angle (in radians) of originating point in the disc of
    point in the jet stream, in the disc plane (counter-clockwise from east)"""
    i = np.radians(90. - inc)
    pa = np.radians(pa)

    # Set up rotation matrices in inclination and position angle, respectively
    rot_x = np.array([[1., 0., 0.],
                      [0., np.cos(i), -np.sin(i)],
                      [0., np.sin(i), np.cos(i)]])
    rot_y = np.array([[np.cos(-pa), 0., np.sin(-pa)],
                      [0., 1., 0.],
                      [-np.sin(-pa), 0., np.cos(-pa)]])

    # Reverse any inclination or rotation
    p = rot_y.dot(rot_x.dot([x, y, z]))

    return np.arctan2(p[1], p[0])


def r_eff(w, R_1, R_2, w_0, r, mod_r_0, r_0, eps):
    return R_1 + (mod_r_0 ** eps * (R_2 - R_1) * w) /\
           (w_0 * (r + mod_r_0 - r_0) ** eps)


def w_r(r, w_0, mod_r_0, r_0, eps):
    return w_0 * ((r + mod_r_0 - r_0) / mod_r_0) ** eps


def y1_y2(x, z, w_0, r_0, mod_r_0, inc):
    i = np.radians(inc)
    y = (np.array([-1., 1]) *
         np.sqrt((-2. * r_0 * w_0 ** 2. * np.cos(i) +
                   2. * mod_r_0 * w_0 ** 2. * np.cos(i) +
                   mod_r_0 ** 2. * z * np.sin(2. * i) +
                   2. * w_0 ** 2. * z * np.sin(i) * np.cos(i)) ** 2. -
                  4. * (w_0 ** 2. * np.cos(i) ** 2. -
                        mod_r_0 ** 2. * np.sin(i) ** 2.) *
                  (-2. * r_0 * w_0 ** 2. * z * np.sin(i) +
                   2. * mod_r_0 * w_0 ** 2. * z * np.sin(i) +
                   mod_r_0 ** 2. * z ** 2. * np.sin(i) ** 2. +
                   w_0 ** 2. * z ** 2. * np.sin(i) ** 2. +
                   r_0 ** 2. * w_0 ** 2. + mod_r_0 ** 2. * w_0 ** 2. -
                   2. * r_0 * mod_r_0 * w_0 ** 2. - mod_r_0 ** 2. * x ** 2. -
                   mod_r_0 ** 2. * z ** 2.)) -
         2. * r_0 * w_0 ** 2. * np.cos(i) +
         2. * mod_r_0 * w_0 ** 2. * np.cos(i) +
         mod_r_0 ** 2. * z * np.sin(2. * i) +
         2. * w_0 ** 2. * z * np.sin(i) * np.cos(i)) /\
        (2. * (mod_r_0 ** 2. * np.sin(i) ** 2. - w_0 ** 2. * np.cos(i) ** 2.))
    return y


def y1_y2_wrapped(w_0, r_0, mod_r_0, inc, bound='lower'):
    """For use with scipy.integrate functions for establishing upper/lower
    bounds in y (first integrated variable)

    Parameters
    ----------
    w_0: float
        Jet half-width
    r_0: float
        Jet ejection radius
    mod_r_0: float
        Modified ejection radius
    inc: float
        Inclination in degrees
    bound: str
        Which bound in the integral to return. One of 'lower' or 'upper'

    Returns
    -------
    Function for use within scipy's integrate methods, for the upper or
    lower-bounds of the first integrated variable, which takes two arguments, z
    and x.
    """
    def func(z, x):
        i = np.radians(inc)
        y = (np.array([-1., 1]) *
             np.sqrt((-2. * r_0 * w_0 ** 2. * np.cos(i) +
                       2. * mod_r_0 * w_0 ** 2. * np.cos(i) +
                       mod_r_0 ** 2. * z * np.sin(2. * i) +
                       2. * w_0 ** 2. * z * np.sin(i) * np.cos(i)) ** 2. -
                      4. * (w_0 ** 2. * np.cos(i) ** 2. -
                            mod_r_0 ** 2. * np.sin(i) ** 2.) *
                      (-2. * r_0 * w_0 ** 2. * z * np.sin(i) +
                       2. * mod_r_0 * w_0 ** 2. * z * np.sin(i) +
                       mod_r_0 ** 2. * z ** 2. * np.sin(i) ** 2. +
                       w_0 ** 2. * z ** 2. * np.sin(i) ** 2. +
                       r_0 ** 2. * w_0 ** 2. + mod_r_0 ** 2. * w_0 ** 2. -
                       2. * r_0 * mod_r_0 * w_0 ** 2. - mod_r_0 ** 2. * x ** 2. -
                       mod_r_0 ** 2. * z ** 2.)) -
             2. * r_0 * w_0 ** 2. * np.cos(i) +
             2. * mod_r_0 * w_0 ** 2. * np.cos(i) +
             mod_r_0 ** 2. * z * np.sin(2. * i) +
             2. * w_0 ** 2. * z * np.sin(i) * np.cos(i)) /\
            (2. * (mod_r_0 ** 2. * np.sin(i) ** 2. - w_0 ** 2. * np.cos(i) ** 2.))
        return y[['lower', 'upper'].index(bound)]
    return func


def w_r_wrapped(w_0, mod_r_0, r_0, eps, inc, bound='lower'):
    """For use with scipy.integrate functions for establishing upper/lower
    bounds in x (second integrated variable). Assumes position angle is 0.

    Parameters
    ----------
    w_0: float
        Jet half-width
    mod_r_0: float
        Modified ejection radius
    r_0: float
        Jet ejection radius
    eps: float
        Power-law index for jet-width as a function of r
    inc: float
        Inclination in degrees
    bound: str
        Which bound in the integral to return. One of 'lower' or 'upper'

    Returns
    -------
    Function for use within scipy's integrate methods, for the upper or
    lower-bounds of the first integrated variable, which takes two arguments, z
    and x.
    """
    def func(z):
        # Assuming position angle is 0.
        i = np.radians(inc)
        r = z / np.sin(i)
        fac = [-1, 1][['lower', 'upper'].index(bound)]
        return w_0 * ((r + mod_r_0 - r_0) / mod_r_0) ** eps * fac
    return func


def xyz_to_rwp(x, y, z, inc, pa):
    i = np.radians(inc)
    t = np.radians(pa)
    r = x * np.sin(i) * np.sin(t) + y * np.cos(i) + z * np.sin(i) * np.cos(t)
    w = np.sqrt(np.sin(i) ** 2. * (y ** 2. - x ** 2. * np.sin(t) ** 2.
                                   - x * z * np.sin(2. * t)
                                   - z ** 2. * np.cos(t) ** 2.)
                - y * np.sin(2. * i) * (x * np.sin(t) + z * np.cos(t))
                + x ** 2. + z ** 2.)
    p = np.arctan2(y * np.sin(i) - z * np.cos(i),
                   x * np.cos(t) - np.sin(t) * (y * np.cos(i) + z * np.sin(i)))
    return r, w, p


def w_xy(x, y, w_0, r_0, eps, opang):
    """
    Compute z-coordinate(s) of jet-boundary point given its x and y
    coordinates.

    Parameters
    ----------
    x : float or Iterable
        x-coordinate(s).
    y : float or Iterable
        y-coordinate(s).
    w_0 : float
        Jet half-width at base.
    r_0 : float
        Jet launching radius.
    eps : float
        Power-law index for jet-width.
    opang : float
        Opening angle at base of jet

    Returns
    -------
    float or numpy.array
        z-coordinate(s) corresponding to supplied x/y coordinate(s) of jet
        boundary.
    """
    # for idx, coord in enumerate([x, y]):
    #     if isinstance(coord, (float, np.floating)):
    #         pass
    #     elif isinstance(coord, (int, np.integer)):
    #         if idx:
    #             y = float(y)
    #         else:
    #             x = float(x)
    #     elif isinstance(coord, Iterable):
    #         if idx:
    #             y = np.array(y, dtype='float')
    #         else:
    #             x = np.array(x, dtype='float')
    #     else:
    #         raise TypeError(["x", "y"][idx] +
    #                         "-coordinate(s) must be float or Iterable")
    mod_r_0 = eps * w_0 / np.tan(np.radians(opang / 2.))

    z = mod_r_0 * ((x**2. + y**2.)**.5 / w_0)**(1. / eps)
    z -= mod_r_0
    z += r_0

    return np.where(z > r_0, z, r_0)


def w_xz(x, z, w_0, r_0, eps, opang):
    """
    Compute y-coordinate(s) of jet-boundary point given its x and z
    coordinates.

    Parameters
    ----------
    x : float or Iterable
        x-coordinate(s).
    z : float or Iterable
        z-coordinate(s).
    w_0 : float
        Jet half-width at base.
    r_0 : float
        Jet launching radius.
    eps : float
        Power-law index for jet-width.
    opang : float
        Opening angle at base of jet

    Returns
    -------
    float or numpy.array
        y-coordinate(s) corresponding to supplied x/z coordinate(s) of jet
        boundary.
    """
    # for idx, coord in enumerate([x, z]):
    #     if isinstance(coord, (float, np.floating)):
    #         pass
    #     elif isinstance(coord, (int, np.integer)):
    #         if idx:
    #             z = float(z)
    #         else:
    #             x = float(x)
    #     elif isinstance(coord, Iterable):
    #         if idx:
    #             x = np.array(x, dtype='float')
    #         else:
    #             z = np.array(z, dtype='float')
    #     else:
    #         raise TypeError(["x", "z"][idx] +
    #                         "-coordinate(s) must be float or Iterable")

    mod_r_0 = eps * w_0 / np.tan(np.radians(opang / 2.))
    y = (w_0**2. * ((z + mod_r_0 - r_0) / mod_r_0)**(2. * eps) - x**2.)
    y = np.abs(y) ** 0.5 * np.sign(y)

    return np.where(z >= r_0, y, 0.)


def w_yz(y, z, w_0, r_0, eps, opang):
    """
    Compute x-coordinate(s) of jet-boundary point given its y and z
    coordinates.

    Parameters
    ----------
    y : float or Iterable
        y-coordinate(s).
    z : float or Iterable
        z-coordinate(s).
    w_0 : float
        Jet half-width at base.
    r_0 : float
        Jet launching radius.
    eps : float
        Power-law index for jet-width.
    opang : float
        Opening angle at base of jet

    Returns
    -------
    float or numpy.array
        x-coordinate(s) corresponding to supplied y/z coordinate(s) of jet
        boundary.
    """
    # for idx, coord in enumerate([y, z]):
    #     if isinstance(coord, (float, np.floating)):
    #         pass
    #     elif isinstance(coord, (int, np.integer)):
    #         if idx:
    #             y = float(y)
    #         else:
    #             z = float(z)
    #     elif isinstance(coord, Iterable):
    #         if idx:
    #             y = np.array(y, dtype='float')
    #         else:
    #             z = np.array(z, dtype='float')
    #     else:
    #         raise TypeError(["y", "z"][idx] +
    #                         "-coordinate(s) must be float or Iterable")

    mod_r_0 = eps * w_0 / np.tan(np.radians(opang / 2.))
    x = (w_0**2. * ((z + mod_r_0 - r_0) / mod_r_0)**(2. * eps) - y**2.)
    x = np.abs(x) ** 0.5 * np.sign(x)

    return np.where(z >= r_0, x, 0.)
