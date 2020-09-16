# -*- coding: utf-8 -*-
"""
Module handling all mathematical functions and methods
"""

import numpy as np
from collections.abc import Iterable


# def w_xy(x, y, w_0, r_0, eps):
#     """
#     Compute z-coordinate(s) of jet-boundary point given its x and y
#     coordinates.
#
#     Parameters
#     ----------
#     x : float or Iterable
#         x-coordinate(s).
#     y : float or Iterable
#         y-coordinate(s).
#     w_0 : float
#         Jet half-width at base.
#     r_0 : float
#         Jet launching radius.
#     eps : float
#         Power-law index for jet-width.
#
#     Returns
#     -------
#     float or numpy.array
#         z-coordinate(s) corresponding to supplied x/y coordinate(s) of jet
#         boundary.
#     """
#     for idx, coord in enumerate([x, y]):
#         if isinstance(coord, (float, np.floating)):
#             pass
#         elif isinstance(coord, (int, np.integer)):
#             if idx:
#                 y = float(y)
#             else:
#                 x = float(x)
#         elif isinstance(coord, Iterable):
#             if idx:
#                 y = np.array(y)
#             else:
#                 x = np.array(x)
#         else:
#             raise TypeError(["x", "y"][idx] +
#                             "-coordinate(s) must be float or Iterable")
#
#     z = r_0 * (np.sqrt(x**2. + y**2.) / w_0)**(1. / eps)
#     return np.where(z > r_0, z, r_0)
#     # if z > r_0:
#     #     return z
#     # else:
#     #     return r_0
#
# def w_xz(x, z, w_0, r_0, eps):
#     """
#     Compute y-coordinate(s) of jet-boundary point given its x and z
#     coordinates.
#
#     Parameters
#     ----------
#     x : float or Iterable
#         x-coordinate(s).
#     z : float or Iterable
#         z-coordinate(s).
#     w_0 : float
#         Jet half-width at base.
#     r_0 : float
#         Jet launching radius.
#     eps : float
#         Power-law index for jet-width.
#
#     Returns
#     -------
#     float or numpy.array
#         y-coordinate(s) corresponding to supplied x/z coordinate(s) of jet
#         boundary.
#     """
#     for idx, coord in enumerate([x, z]):
#         if isinstance(coord, (float, np.floating)):
#             pass
#         elif isinstance(coord, (int, np.integer)):
#             if idx:
#                 z = float(z)
#             else:
#                 x = float(x)
#         elif isinstance(coord, Iterable):
#             if idx:
#                 x = np.array(x)
#             else:
#                 z = np.array(z)
#         else:
#             raise TypeError(["x", "z"][idx] +
#                             "-coordinate(s) must be float or Iterable")
#     y = np.sqrt(w_0**2. * (z / r_0)**(2. * eps) - x**2.)
#     return y
#
# def w_yz(y, z, w_0, r_0, eps):
#     """
#     Compute x-coordinate(s) of jet-boundary point given its y and z
#     coordinates.
#
#     Parameters
#     ----------
#     y : float or Iterable
#         y-coordinate(s).
#     z : float or Iterable
#         z-coordinate(s).
#     w_0 : float
#         Jet half-width at base.
#     r_0 : float
#         Jet launching radius.
#     eps : float
#         Power-law index for jet-width.
#
#     Returns
#     -------
#     float or numpy.array
#         x-coordinate(s) corresponding to supplied y/z coordinate(s) of jet
#         boundary.
#     """
#     for idx, coord in enumerate([y, z]):
#         if isinstance(coord, (float, np.floating)):
#             pass
#         elif isinstance(coord, (int, np.integer)):
#             if idx:
#                 y = float(y)
#             else:
#                 z = float(z)
#         elif isinstance(coord, Iterable):
#             if idx:
#                 y = np.array(y)
#             else:
#                 z = np.array(z)
#         else:
#             raise TypeError(["y", "z"][idx] +
#                             "-coordinate(s) must be float or Iterable")
#     x = np.sqrt(w_0**2. * (z / r_0)**(2. * eps) - y**2.)
#     return x

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

    z = mod_r_0 * (np.sqrt(x**2. + y**2.) / w_0)**(1. / eps)
    z -= mod_r_0
    z += r_0

    return np.where(z >= r_0, z, r_0)


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
    y = np.sqrt(w_0**2. * ((z + mod_r_0 - r_0) / mod_r_0)**(2. * eps) - x**2.)

    return np.where(z >= r_0, y, np.NaN)


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
    x = np.sqrt(w_0**2. * ((z + mod_r_0 - r_0) / mod_r_0)**(2. * eps) - y**2.)

    return np.where(z >= r_0, x, np.NaN)
