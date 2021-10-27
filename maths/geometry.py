# -*- coding: utf-8 -*-
"""
Module handling all mathematical functions and methods
"""
from typing import Union
import numpy as np
from collections.abc import Iterable
import scipy.constants as con
from scipy.special import hyp2f1


def mod_r_0(opang: float, epsilon: float, w_0: float) -> float:
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


def rho(r: Union[float, Iterable], r_0: float,
        mr0: Union[None, float] = None) -> Union[float, Iterable]:
    """
    Calculates distance along r-axis in units of r_0 in the case whereby
    mod_r_0 is None (default, as per Reynolds 1986). In the case mod_r_0 is
    not None, the equivalent distance along hte r-axis in units of mod_r_0,
    when r is translated by a factor of (mod_r_0 - r_0). All input arg units
    should be consistent

    Parameters
    ----------
    r: float
        Jet radius coordinate
    r_0 : float
        Launching radius
    mr0 : float
        Reynolds (1986)'s value for r_0 given specified geometry (see
        RaJePy.maths.geometry.mod_r_0 method)

    Returns
    -------
    Distance of r along r-axis in units of r_0 (if mod_r_0 arg is None) or
    units of mod_r_0 after a translation of r by a factor of (mod_r_0 - r_0)
    """
    if mr0:
        return (np.abs(r) + mr0 - r_0) / mr0
    else:
        return np.abs(r) / r_0


def cell_value(zero_val: float, rho: Union[float, Iterable],
               reff: Union[float, Iterable], r1: float, q: float,
               qd: float) -> Union[float, Iterable]:
    """
    Equation to calculate a cell's physical parameter value. Make sure units for
    args are all consistent.

    Parameters
    ----------
    zero_val
        Value of physical parameter at r=0, w=0
    rho
        Distance along r-axis in units of r_0
    reff
        Radius in the accretion disc at which the cell's material was sources
    r1
        Inner radius of disc for launch of material
    q
        Power-law exponent for the behaviour of the physical quantity as a
        function of rho
    qd
        Power-law exponent for the behaviour of the physical quantity as a
        function of (reff / r1)

    Returns
    -------
    Value of physical quantity in cell
    """
    vals = zero_val * rho ** q * (reff / r1) ** qd
    return vals

def w_r(r: Union[float, Iterable], w_0: float, mr0: float, r_0: float,
        eps: float) -> Union[float, Iterable]:
    """
    Jet-width, w(r),  as a function of r, the distance along the jet axis

    Parameters
    ----------
    r : float or np.ndarray
        Distance(s) along jet axis at which to compute w
    w_0 : float
        Width of the jet at its base
    mr0 : float
        Reynolds (1986)'s value for r_0 given specified geometry (see
        RaJePy.maths.geometry.mod_r_0 method)
    r_0 : float
        Launching radius
    eps : float
        Power-law exponent for the growth of w with r
    Returns
    -------
    Jet-width at r
    """
    return w_0 * rho(r, r_0, mr0) ** eps


def t_rw(r: Union[float, Iterable, np.ndarray],
         w: Union[float, Iterable, np.ndarray],
         params: dict) -> Union[float, Iterable, np.ndarray]:
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
    mr0 = params['geometry']['mod_r_0'] * con.au
    eps = params['geometry']['epsilon']
    r_1 = params["target"]["R_1"] * con.au
    r_2 = params["target"]["R_2"] * con.au
    q_v = params["power_laws"]["q_v"]
    q_vd = params["power_laws"]["q^d_v"]

    def indef_intgrl(_r, _w):
        """
        Indefinite integral in r
        """
        const = mr0 ** q_v / (v_0 * (1. - q_v + eps * q_vd))
        rad = _r + mr0 - r_0
        p1 = rad ** (1. - q_v)
        p2 = (r_eff(_w, r_1, r_2, w_0, _r, mr0, r_0, eps) / r_1) ** -q_vd

        # To avoid ZeroDivisionError when w = 0
        if _w == 0.:
            p3 = 1.0
            # Factor of 1. + q_vd / (1. - q_v) introduced here as computed times
            # are always too large at w = 0
            p4 = 1. + q_vd / (1. - q_v)
        else:
            p3 = ((r_1 * w_0 * rad ** eps) /
                  (_w * mr0 ** eps * (r_2 - r_1)) + 1.) ** q_vd
            p4 = hyp2f1(q_vd, (1. - q_v + eps * q_vd) / eps,
                        (1. - q_v + eps + eps * q_vd) / eps,
                        (r_1 * w_0 * rad ** eps) /
                        (_w * mr0 ** eps * (r_1 - r_2)))

        return const * p1 * p2 * p3 * p4

    indef_intgrl = np.vectorize(indef_intgrl)

    return (indef_intgrl(np.abs(r) * con.au, w * con.au) -
            indef_intgrl(r_0, w * con.au)) / con.year


def xyz_to_rwp(x: Union[float, Iterable],
               y: Union[float, Iterable],
               z: Union[float, Iterable],
               inc: float, pa: float) -> Union[float, Iterable]:
    """
    Converts (x, y, z) coordinate to jet-system coordinates (r, w, phi)

    Parameters
    ----------
    x : float
        x-coordinate
    y : float
        y-coordinate
    z : float
        z-coordinate
    inc : float
        Inclination of system (deg)
    pa : float
        Position angle of system (deg)

    Returns
    -------
    Tuple of r, w, and phi coordinate arrays
    """
    xyz_derotated = xyz_rotate(x, y, z, inc - 90., pa, order='yx')
    w, p, r = cartesian_to_cylindrical(*xyz_derotated)

    return r, w, p


def xyz_rotate(x: Union[float, Iterable],
               y: Union[float, Iterable],
               z: Union[float, Iterable],
               alpha: float, beta: float,
               order: str = 'xy') -> Union[float, Iterable]:
    """
    Converts rotated (x, y, z) coordinate(s) to derotated (x, y, z)
    coordinate(s) given a rotation of defined x-rotation and y-rotation.
    See https://en.wikipedia.org/wiki/Rotation_matrix.

    Parameters
    ----------
    x
        x-coordinate(s)
    y
        y-coordinate(s)
    z
        z-coordinate(s)
    alpha
        Rotation angle around x-axis (right-handed), deg
    beta
        Rotation angle around y-axis (right-handed), deg
    order
        Order in which to rotate. Default is 'xy' indicating rotate around
        the x-axis first, the y-axis second. 'yx' only other legal value.

    Returns
    -------
    Rotated (x, y, z) coordinate
    """
    a = np.radians(alpha)
    b = np.radians(beta)

    cos_a, sin_a = np.cos(a), np.sin(a)
    cos_b, sin_b = np.cos(b), np.sin(b)

    def x_rot(x, y, z):
        """x-rotation matrix's dot product with (x, y, z)"""
        return x, cos_a * y - sin_a * z, sin_a * y + cos_a * z

    def y_rot(x, y, z):
        """y-rotation matrix's dot product with (x, y, z)"""
        return cos_b * x + sin_b * z, y, cos_b * z - sin_b * x

    if order.lower() == 'xy':
        return y_rot(*x_rot(x, y, z))
    elif order.lower() == 'yx':
        return x_rot(*y_rot(x, y, z))
    else:
        raise ValueError(f"Order of rotation, {order.__repr__()}, not "
                         "recognised")


def cartesian_to_cylindrical(x: Union[float, Iterable],
                             y: Union[float, Iterable],
                             z: Union[float, Iterable]) -> Union[float, Iterable]:
    """
    Converts NON-rotated Cartesian (x, y, z) coordinate(s) to cylindrical
    (rho, phi, z) coordinates.
    See https://en.wikipedia.org/wiki/Cylindrical_coordinate_system.

    Parameters
    ----------
    x
        x-coordinate(s)
    y
        y-coordinate(s)
    z
        z-coordinate(s)

    Returns
    -------
    Cylindrical (rho, phi, z) coordinate(s)
    """
    _rho = np.sqrt(x ** 2. + y ** 2.)
    _phi = np.arcsin(y / _rho)

    if isinstance(x, Iterable):
        _phi = np.where(x < 0, -_phi + np.pi, _phi)
    elif isinstance(x, float) or isinstance(x, int):
        if x < 0:
            _phi = -_phi + np.pi
    else:
        raise TypeError(f"x-coordinate is of type-{type(x)}, but should be a "
                        "float or iterable")

    return _rho, _phi, z


def r_eff(w: Union[float, np.ndarray], r_1: float, r_2: float, w_0: float,
          r: Union[float, np.ndarray], mr0: float, r_0: float,
          eps: float) -> Union[float, np.ndarray]:
    """
    Radius in the accretion disc at which the material at jet-coordinates (w, r)
    was ejected from

    Parameters
    ----------
    w : float, Iterable
        Jet w-coordinate
    r_1 : float
        Inner radius of disc for launch of material
    r_2 : float
        Inner radius of disc for launch of material
    w_0 : float
        Width of the jet at its base
    r : float, Iterable
        Jet r-coordinate
    mr0 : float
        Reynolds (1986)'s value for r_0 given specified geometry (see
        RaJePy.maths.geometry.mod_r_0 method)
    r_0 : float
        Launching radius
    eps : float
        Power-law exponent for the growth of w with r

    Returns
    -------
    Effective radius at coordinate (w, r)
    """
    return r_1 + ((r_2 - r_1) * w) / w_r(r, w_0, mr0, r_0, eps)

# def disc_angle(x: Union[float, Iterable],
#                y: Union[float, Iterable],
#                z: Union[float, Iterable],
#                inc: float, pa: float) -> Union[float, np.ndarray]:
#     """
#     Returns position angle (in radians) of originating point in the disc of
#     point in the jet stream, in the disc plane (counter-clockwise from east)
#     """
#     i = np.radians(90. - inc)
#     pa = np.radians(pa)
#
#     # Set up rotation matrices in inclination and position angle, respectively
#     rot_x = np.array([[1., 0., 0.],
#                       [0., np.cos(i), -np.sin(i)],
#                       [0., np.sin(i), np.cos(i)]])
#     rot_y = np.array([[np.cos(-pa), 0., np.sin(-pa)],
#                       [0., 1., 0.],
#                       [-np.sin(-pa), 0., np.cos(-pa)]])
#
#     # Reverse any inclination or rotation
#     p = rot_y.dot(rot_x.dot([x, y, z]))
#
#     return np.arctan2(p[1], p[0])
#
#
#
# def y1_y2(x: Union[float, Iterable], z: Union[float, Iterable], w_0: float,
#           r_0: float, mr0: float, inc: float) -> Union[float, Iterable]:
#     i = np.radians(inc)
#     y = (np.array([-1., 1]) *
#          np.sqrt((-2. * r_0 * w_0 ** 2. * np.cos(i) +
#                   2. * mr0 * w_0 ** 2. * np.cos(i) +
#                   mr0 ** 2. * z * np.sin(2. * i) +
#                   2. * w_0 ** 2. * z * np.sin(i) * np.cos(i)) ** 2. -
#                  4. * (w_0 ** 2. * np.cos(i) ** 2. -
#                        mr0 ** 2. * np.sin(i) ** 2.) *
#                  (-2. * r_0 * w_0 ** 2. * z * np.sin(i) +
#                   2. * mr0 * w_0 ** 2. * z * np.sin(i) +
#                   mr0 ** 2. * z ** 2. * np.sin(i) ** 2. +
#                   w_0 ** 2. * z ** 2. * np.sin(i) ** 2. +
#                   r_0 ** 2. * w_0 ** 2. + mr0 ** 2. * w_0 ** 2. -
#                   2. * r_0 * mr0 * w_0 ** 2. - mr0 ** 2. * x ** 2. -
#                   mr0 ** 2. * z ** 2.)) -
#          2. * r_0 * w_0 ** 2. * np.cos(i) +
#          2. * mr0 * w_0 ** 2. * np.cos(i) +
#          mr0 ** 2. * z * np.sin(2. * i) +
#          2. * w_0 ** 2. * z * np.sin(i) * np.cos(i)) / \
#         (2. * (mr0 ** 2. * np.sin(i) ** 2. - w_0 ** 2. * np.cos(i) ** 2.))
#     return y
#
#
# def y1_y2_wrapped(w_0: float, r_0: float, mr0: float, inc: float,
#                   bound: str = 'lower') -> Callable:
#     """
#     For use with scipy.integrate functions for establishing upper/lower
#     bounds in y (first integrated variable)
#
#     Parameters
#     ----------
#     w_0: float
#         Jet half-width
#     r_0: float
#         Jet ejection radius
#     mr0: float
#         Modified ejection radius
#     inc: float
#         Inclination in degrees
#     bound: str
#         Which bound in the integral to return. One of 'lower' or 'upper'
#
#     Returns
#     -------
#     Function for use within scipy's integrate methods, for the upper or
#     lower-bounds of the first integrated variable, which takes two arguments, z
#     and x.
#     """
#
#     def func(z, x):
#         i = np.radians(inc)
#         y = (np.array([-1., 1]) *
#              np.sqrt((-2. * r_0 * w_0 ** 2. * np.cos(i) +
#                       2. * mr0 * w_0 ** 2. * np.cos(i) +
#                       mr0 ** 2. * z * np.sin(2. * i) +
#                       2. * w_0 ** 2. * z * np.sin(i) * np.cos(i)) ** 2. -
#                      4. * (w_0 ** 2. * np.cos(i) ** 2. -
#                            mr0 ** 2. * np.sin(i) ** 2.) *
#                      (-2. * r_0 * w_0 ** 2. * z * np.sin(i) +
#                       2. * mr0 * w_0 ** 2. * z * np.sin(i) +
#                       mr0 ** 2. * z ** 2. * np.sin(i) ** 2. +
#                       w_0 ** 2. * z ** 2. * np.sin(i) ** 2. +
#                       r_0 ** 2. * w_0 ** 2. + mr0 ** 2. * w_0 ** 2. -
#                       2. * r_0 * mr0 * w_0 ** 2. - mr0 ** 2. * x ** 2. -
#                       mr0 ** 2. * z ** 2.)) -
#              2. * r_0 * w_0 ** 2. * np.cos(i) +
#              2. * mr0 * w_0 ** 2. * np.cos(i) +
#              mr0 ** 2. * z * np.sin(2. * i) +
#              2. * w_0 ** 2. * z * np.sin(i) * np.cos(i)) / \
#             (2. * (mr0 ** 2. * np.sin(i) ** 2. - w_0 ** 2. * np.cos(
#                 i) ** 2.))
#         return y[['lower', 'upper'].index(bound)]
#
#     return func
#
#
# def w_r_wrapped(w_0: float, mr0: float, r_0: float, eps: float, inc: float,
#                 bound: str = 'lower') -> Callable:
#     """
#     For use with scipy.integrate functions for establishing upper/lower
#     bounds in x (second integrated variable). Assumes position angle is 0.
#
#     Parameters
#     ----------
#     w_0: float
#         Jet half-width
#     mr0: float
#         Modified ejection radius
#     r_0: float
#         Jet ejection radius
#     eps: float
#         Power-law index for jet-width as a function of r
#     inc: float
#         Inclination in degrees
#     bound: str
#         Which bound in the integral to return. One of 'lower' or 'upper'
#
#     Returns
#     -------
#     Function for use within scipy's integrate methods, for the upper or
#     lower-bounds of the first integrated variable, which takes two arguments, z
#     and x.
#     """
#
#     def func(z: Union[float, Iterable]) -> Union[float, Iterable]:
#         # Assuming position angle is 0.
#         i = np.radians(inc)
#         r = z / np.sin(i)
#         fac = [-1, 1][['lower', 'upper'].index(bound)]
#         return w_r(r, w_0, mr0, r_0, eps) * fac
#
#     return func
#
#
# def w_xy(x: Union[float, Iterable], y: Union[float, Iterable], w_0: float,
#          r_0: float, eps: float, opang: float) -> Union[float, Iterable]:
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
#     opang : float
#         Opening angle at base of jet
#
#     Returns
#     -------
#     float or numpy.array
#         z-coordinate(s) corresponding to supplied x/y coordinate(s) of jet
#         boundary.
#     """
#     mr0 = mod_r_0(opang, eps, w_0)
#
#     z = mr0 * ((x ** 2. + y ** 2.) ** .5 / w_0) ** (1. / eps)
#     z -= mr0
#     z += r_0
#
#     return np.where(z > r_0, z, r_0)
#
#
# def w_xz(x: Union[float, Iterable], z: Union[float, Iterable], w_0: float,
#          r_0: float, eps: float, opang: float) -> Union[float, Iterable]:
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
#     opang : float
#         Opening angle at base of jet
#
#     Returns
#     -------
#     float or numpy.array
#         y-coordinate(s) corresponding to supplied x/z coordinate(s) of jet
#         boundary.
#     """
#     y = (w_r(z, w_0, mod_r_0(opang, eps, w_0), r_0, eps) ** 2. - x ** 2.)
#     y = np.abs(y) ** 0.5 * np.sign(y)
#
#     return np.where(z >= r_0, y, 0.)
#
#
# def w_yz(y: Union[float, Iterable], z: Union[float, Iterable], w_0: float,
#          r_0: float, eps: float, opang: float) -> Union[float, Iterable]:
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
#     opang : float
#         Opening angle at base of jet
#
#     Returns
#     -------
#     float or numpy.array
#         x-coordinate(s) corresponding to supplied y/z coordinate(s) of jet
#         boundary.
#     """
#     x = (w_r(z, w_0, mod_r_0(opang, eps, w_0), r_0, eps) ** 2. - y ** 2.)
#     x = np.abs(x) ** 0.5 * np.sign(x)
#
#     return np.where(z >= r_0, x, 0.)
#
#
# # Deprecated functions below
# def r_xyzti(x: Union[float, Iterable],
#             y: Union[float, Iterable],
#             z: Union[float, Iterable],
#             inc: float, pa: float) -> Union[float, Iterable]:
#     """
#     Converts (x, y, z) coordinate to jet-system r-coordinate(s) given a jet of
#     defined inclination and position angle.
#
#     Parameters
#     ----------
#     x : float
#         x-coordinate
#     y : float
#         y-coordinate
#     z : float
#         z-coordinate
#     inc : float
#         Inclination of system (deg)
#     pa : float
#         Position angle of system (deg)
#
#     Returns
#     -------
#     r-coordinate(s) of given (x, y, z) coordinate
#     """
#     i = np.radians(-inc)  # Blue jet is subsequently inclined towards us
#     t = np.radians(-pa)   # East of North in an RA/Dec. sense
#     r = x * np.sin(i) * np.sin(t) + y * np.cos(i) + z * np.sin(i) * np.cos(t)
#     # r = np.cos(t) * (y * np.cos(i) + z * np.sin(i)) - x * np.sin(t)  # Comes from rotating by rot_x, then rot_y
#     # r = y * np.cos(i) + np.sin(i) * (z * np.cos(t) - x * np.sin(t))  # Comes from derotating by rot_y, then rot_x
#
#     return r
#
#
# def w_xyzti(x: Union[float, Iterable],
#             y: Union[float, Iterable],
#             z: Union[float, Iterable],
#             inc: float, pa: float) -> Union[float, Iterable]:
#     """
#     Converts (x, y, z) coordinate to jet-system w-coordinate(s) given a jet of
#     defined inclination and position angle.
#
#     Parameters
#     ----------
#     x : float
#         x-coordinate
#     y : float
#         y-coordinate
#     z : float
#         z-coordinate
#     inc : float
#         Inclination of system (deg)
#     pa : float
#         Position angle of system (deg)
#
#     Returns
#     -------
#     w-coordinate(s) of given (x, y, z) coordinate
#     """
#     i = np.radians(-inc)  # Blue jet is subsequently inclined towards us
#     t = np.radians(-pa)   # East of North in an RA/Dec. sense
#     w = np.sqrt(np.sin(i) ** 2. * (y ** 2. - x ** 2. * np.sin(t) ** 2.
#                                    - x * z * np.sin(2. * t)
#                                    - z ** 2. * np.cos(t) ** 2.)
#                 - y * np.sin(2. * i) * (x * np.sin(t) + z * np.cos(t))
#                 + x ** 2. + z ** 2.)
#     # cos_t, sin_t = np.cos(t), np.cos(t)
#     # cos_i, sin_i = np.cos(i), np.sin(i)
#     # w = np.sqrt((x * cos_t + sin_t * (y * cos_i + z * sin_i)) ** 2. +
#     #             (y * sin_i - z * cos_i) ** 2.)
#     # w = np.sqrt((x * cos_t + z * sin_t) ** 2. +
#     #             (y * sin_i - cos_i * (z * cos_t - x * sin_t)) ** 2.)
#
#     return w
#
#
# def phi_xyzti(x: Union[float, Iterable],
#               y: Union[float, Iterable],
#               z: Union[float, Iterable],
#               inc: float, pa: float) -> Union[float, Iterable]:
#     """
#     Converts (x, y, z) coordinate to jet-system phi-coordinate(s) given a jet of
#     defined inclination and position angle.
#
#     Parameters
#     ----------
#     x : float
#         x-coordinate(s)
#     y : float
#         y-coordinate(s)
#     z : float
#         z-coordinate(s)
#     inc : float
#         Inclination of system (deg)
#     pa : float
#         Position angle of system (deg)
#
#     Returns
#     -------
#     phi-coordinate(s) of given (x, y, z) coordinate
#     """
#     i = np.radians(-inc)  # Blue jet is subsequently inclined towards us
#     t = np.radians(-pa)   # East of North in an RA/Dec. sense
#
#     xr = x * np.cos(t) - np.sin(t) * (y * np.cos(i) + z * np.sin(i))
#     phi = np.arctan2(y * np.sin(i) - z * np.cos(i), xr)
#
#     # w = w_xyzti(x, y, z, inc, pa)
#     # phi = np.arcsin((y * np.sin(i) - z * np.cos(i)) / w)
#     # phi = np.arcsin((y * np.sin(i) - np.cos(i) * (z * np.cos(t) - x * np.sin(t))) / w)
#
#     if isinstance(x, Iterable):
#         phi = np.where(xr < 0, -phi + np.pi, phi)
#     elif isinstance(xr, float) or isinstance(xr, int):
#         if xr < 0:
#             phi = -phi + np.pi
#     else:
#         raise TypeError(f"x-coordinate is of type-{type(x)}, but should be a "
#                         "float or iterable")
#
#     return phi
