# -*- coding: utf-8 -*-
"""
Module handling all mathematical functions and methods
"""
from typing import Union, Tuple
import numpy as np
from collections.abc import Iterable
import scipy.constants as con
from scipy.special import hyp2f1
import dask.array as da
from ..geometry import cell_value, mod_r_0, r_eff, w_r

# Only need to change type annotations for these functions
for arg in ('rho_', 'r_eff_', 'return', 'w', 'r'):
    if arg in cell_value.__annotations__:
        cell_value.__annotations__[arg] = Union[float, da.core.Array]
    if arg in w_r.__annotations__:
        w_r.__annotations__[arg] = Union[float, da.core.Array]
    if arg in r_eff.__annotations__:
        r_eff.__annotations__[arg] = Union[float, da.core.Array]


def rho(r: Union[float, da.core.Array], r_0: float,
        mr0: Union[None, float] = None) -> Union[float, da.core.Array]:
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
        return (da.abs(r) + mr0 - r_0) / mr0
    else:
        return da.abs(r) / r_0


def t_rw(r: Union[float, da.core.Array],
         w: Union[float, da.core.Array],
         params: dict) -> Union[float, da.core.Array]:
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

        p3_mul_p4 = da.where(
            w_ == 0., 1. + q_vd / (1. - q_v),
            ((r_1 * w_0 * rad ** eps) /
                  (w_ * mr0 ** eps * (r_2 - r_1)) + 1.) ** q_vd *
            hyp2f1(q_vd, (1. - q_v + eps * q_vd) / eps,
                        (1. - q_v + eps + eps * q_vd) / eps,
                        (r_1 * w_0 * rad ** eps) /
                        (w_ * mr0 ** eps * (r_1 - r_2)))
        )

        return const * p1 * p2 * p3_mul_p4

    return (indef_intgrl(da.abs(r) * con.au, w * con.au) -
            indef_intgrl(r_0, w * con.au)) / con.year


def xyz_to_rwp(x: Union[float, da.core.Array],
               y: Union[float, da.core.Array],
               z: Union[float, da.core.Array],
               inc: float, pa: float
               ) -> Union[float, Tuple[da.core.Array, da.core.Array, da.core.Array]]:
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


def xyz_rotate(x: Union[float, da.core.Array],
               y: Union[float, da.core.Array],
               z: Union[float, da.core.Array],
               alpha: float, beta: float,
               order: str = 'xy'
               ) -> Union[float, Tuple[da.core.Array,
                                       da.core.Array,
                                       da.core.Array]]:
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
    a = da.radians(alpha)
    b = da.radians(beta)

    cos_a, sin_a = da.cos(a), da.sin(a)
    cos_b, sin_b = da.cos(b), da.sin(b)

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


def cartesian_to_cylindrical(x: Union[float, da.core.Array],
                             y: Union[float, da.core.Array],
                             z: Union[float, da.core.Array]
                             ) -> Union[float, Tuple[da.core.Array,
                                                     da.core.Array,
                                                     da.core.Array]]:
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
    rho_ = da.sqrt(x ** 2. + y ** 2.)
    phi_ = da.arcsin(y / rho_)

    if not np.isscalar(x):
        phi_ = da.where(x < 0, -phi_ + np.pi, phi_)
    elif isinstance(x, (float, int)):
        if x < 0:
            phi_ = -phi_ + np.pi
    else:
        raise TypeError(f"x-coordinate is of type-{type(x)}, but should be a "
                        "float or numpy.typing.ArrayLike")

    return rho_, phi_, z
