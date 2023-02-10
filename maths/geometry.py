# -*- coding: utf-8 -*-
"""
Module handling all mathematical functions and methods
"""
from typing import Union, Tuple
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


def rho(r: Union[float, np.ndarray], r_0: float,
        mr0: Union[None, float] = None) -> Union[float, np.ndarray]:
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


def cell_value(zero_val: float, rho_: Union[float, np.ndarray],
               r_eff_: Union[float, np.ndarray], r1: float, q: float,
               qd: float) -> Union[float, np.ndarray]:
    """
    Equation to calculate a cell's physical parameter value. Make sure units for
    args are all consistent.

    Parameters
    ----------
    zero_val
        Value of physical parameter at r=0, w=0
    rho_
        Distance along r-axis in units of r_0
    r_eff_
        Radius in the accretion disc at which the cell's material was sources
    r1
        Inner radius of disc for launch of material
    q
        Power-law exponent for the behaviour of the physical quantity as a
        function of rho_
    qd
        Power-law exponent for the behaviour of the physical quantity as a
        function of (r_eff_ / r1)

    Returns
    -------
    Value of physical quantity in cell
    """
    vals = zero_val * rho_ ** q * (r_eff_ / r1) ** qd
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


def t_rw(r: Union[float, np.ndarray],
         w: Union[float, np.ndarray],
         params: dict) -> Union[float, np.ndarray]:
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

    def indef_intgrl(r_, w_):
        """
        Indefinite integral in r
        """
        const = mr0 ** q_v / (v_0 * (1. - q_v + eps * q_vd))
        rad = r_ + mr0 - r_0
        p1 = rad ** (1. - q_v)
        p2 = (r_eff(w_, r_1, r_2, w_0, r_, mr0, r_0, eps) / r_1) ** -q_vd

        # To avoid ZeroDivisionError when w = 0
        if w_ == 0.:
            p3 = 1.0
            # Factor of 1. + q_vd / (1. - q_v) introduced here as computed times
            # are always too large at w = 0
            p4 = 1. + q_vd / (1. - q_v)
        else:
            p3 = ((r_1 * w_0 * rad ** eps) /
                  (w_ * mr0 ** eps * (r_2 - r_1)) + 1.) ** q_vd
            p4 = hyp2f1(q_vd, (1. - q_v + eps * q_vd) / eps,
                        (1. - q_v + eps + eps * q_vd) / eps,
                        (r_1 * w_0 * rad ** eps) /
                        (w_ * mr0 ** eps * (r_1 - r_2)))

        return const * p1 * p2 * p3 * p4

    indef_intgrl = np.vectorize(indef_intgrl)

    return (indef_intgrl(np.abs(r) * con.au, w * con.au) -
            indef_intgrl(r_0, w * con.au)) / con.year


def xyz_to_rwp(x: Union[float, np.ndarray],
               y: Union[float, np.ndarray],
               z: Union[float, np.ndarray],
               inc: float, pa: float
               ) -> Union[float, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Converts (x, y, z) coordinate to jet-system coordinates (r, w, phi)

    Parameters
    ----------
    x : float
        x-coordinate [au]
    y : float
        y-coordinate [au]
    z : float
        z-coordinate [au]
    inc : float
        Inclination of system [deg]
    pa : float
        Position angle of system [deg]

    Returns
    -------
    Tuple of r, w, and phi coordinate arrays
    """
    xyz_derotated = xyz_rotate(x, y, z, inc - 90., pa, order='yx')
    w, p, r = cartesian_to_cylindrical(*xyz_derotated)

    return r, w, p


def xyz_rotate(x: Union[float, np.ndarray],
               y: Union[float, np.ndarray],
               z: Union[float, np.ndarray],
               alpha: float, beta: float,
               order: str = 'xy'
               ) -> Union[float, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
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
        the x-axis first, the y-axis second. 'yx' only other allowed value.

    Returns
    -------
    Rotated (x, y, z) coordinate
    """
    a = np.radians(alpha)
    b = np.radians(beta)

    cos_a, sin_a = np.cos(a), np.sin(a)
    cos_b, sin_b = np.cos(b), np.sin(b)

    def x_rot(x_, y_, z_):
        """x-rotation matrix's dot product with (x, y, z)"""
        return x_, cos_a * y_ - sin_a * z_, sin_a * y_ + cos_a * z_

    def y_rot(x_, y_, z_):
        """y-rotation matrix's dot product with (x, y, z)"""
        return cos_b * x_ + sin_b * z_, y_, cos_b * z_ - sin_b * x_

    if order.lower() == 'xy':
        return y_rot(*x_rot(x, y, z))
    elif order.lower() == 'yx':
        return x_rot(*y_rot(x, y, z))
    else:
        raise ValueError(f"Order of rotation, {order.__repr__()}, not "
                         "recognised")


def cartesian_to_cylindrical(x: Union[float, np.ndarray],
                             y: Union[float, np.ndarray],
                             z: Union[float, np.ndarray]
                             ) -> Union[float, Tuple[np.ndarray,
                                                     np.ndarray,
                                                     np.ndarray]]:
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
    rho_ = np.sqrt(x ** 2. + y ** 2.)
    phi_ = np.arcsin(y / rho_)

    if not np.isscalar(x):
        phi_ = np.where(x < 0, -phi_ + np.pi, phi_)
    elif isinstance(x, (float, int)):
        if x < 0:
            phi_ = -phi_ + np.pi
    else:
        raise TypeError(f"x-coordinate is of type-{type(x)}, but should be a "
                        "float or numpy.typing.ArrayLike")

    return rho_, phi_, z


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
