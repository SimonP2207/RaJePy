import os
from scipy.integrate import quad, tplquad, trapz, cumtrapz
import matplotlib.pylab as plt
import numpy as np
from RaJePy.classes import Cell, Grid, JetModel, cfg
from RaJePy.maths import physics as mphys
import scipy.constants as con

def y1_y2(x, z, inc, w_0, r_m, r_0):
    i = np.radians(inc)
    alpha = z * np.cos(i)
    beta = z * np.sin(i) + r_m * np.sign(z) - r_0 * np.sign(z)
    delta = (w_0 / r_m) ** 2.
    y1 = (-np.sqrt(
        alpha ** 2 * delta * np.cos(i) ** 2 + 2 * alpha * beta * delta * np.sin(
            i) * np.cos(i) + beta ** 2 * delta * np.sin(
            i) ** 2 + delta * x ** 2 * np.cos(i) ** 2 - x ** 2 * np.sin(
            i) ** 2) + alpha * np.sin(i) + beta * delta * np.cos(i)) / (
                    np.sin(i) ** 2 - delta * np.cos(i) ** 2)
    y2 = (np.sqrt(
        alpha ** 2 * delta * np.cos(i) ** 2 + 2 * alpha * beta * delta * np.sin(
            i) * np.cos(i) + beta ** 2 * delta * np.sin(
            i) ** 2 + delta * x ** 2 * np.cos(i) ** 2 - x ** 2 * np.sin(
            i) ** 2) + alpha * np.sin(i) + beta * delta * np.cos(i)) / (
                    np.sin(i) ** 2 - delta * np.cos(i) ** 2)
    return y1, y2

def quad_formula(a, b, c):
    discrim = b ** 2. - 4. * a * c
    denom = 2. * a

    return -b - np.sqrt(discrim) / denom, -b + np.sqrt(discrim) / denom

def y1_y2_defunct(x, z, inc, w_0, r_m, r_0):
    """Determines y-coordinates where line of sight intercepts jet boundary
    for the special case of a conical jet (epsilon = 1)"""
    i = np.radians(inc)
    alpha = z * np.cos(i)
    beta = z * np.sin(i) + r_m * np.sign(z) - r_0 * np.sign(z)
    delta = (w_0 / r_m) ** 2.

    a = np.sin(i) ** 2. - delta * np.cos(i) ** 2.
    b = -2. * beta * delta * np.cos(i) - 2. * alpha * np.sin(i)
    c = x ** 2. + alpha ** 2. - delta * beta ** 2.

    return quad_formula(a, b, c)

def xyz_to_rw(x, y, z, inc):
    i = np.radians(inc)
    r = y * np.cos(i) + z * np.sin(i)
    w = np.sqrt(x ** 2. + (z * np.cos(i) - y * np.sin(i)) ** 2.)
    return r, w

def xyz_to_rw(x, y, z, inc, pa):
    i = np.radians(inc)
    t = np.radians(pa)
    r = x * np.sin(i) * np.sin(t) + y * np.cos(i) + z * np.sin(i) * np.cos(t)
    w = np.sqrt(np.sin(i) ** 2. * (-x ** 2. * np.sin(t) ** 2. - x * z *
                                   np.sin(2. * t) + y ** 2. - z ** 2. *
                                   np.cos(t) ** 2.)
                - y * np.sin(2. * i) * (x * np.sin(t) + z * np.cos(t)) + x ** 2.
                + z ** 2.)
    return r, w

def ne_y_func(x, z, n_0, x_0, w_0, r_m, r_0, r_1, r_2, q_n, q_nd, q_x, q_xd,
              eps, inc):
    def func(y):
        r, w = xyz_to_rw(x, y, z, inc)
        rho = (np.abs(r) + r_m - r_0) / r_m
        w_r = w_0 * rho ** eps

        if np.abs(r) < r_0:
            return 0.
        if w > w_r:
            return 0.

        p1 = x_0 * n_0 * (1. + ((r_2 - r_1) * w) /
             (r_1 * w_0 * rho ** eps)) ** (q_nd + q_xd)
        p2 = rho ** (q_n + q_x)

        return p1 * p2
    return np.vectorize(func)

def T_y_func(x, z, T_0, w_0, r_m, r_0, r_1, r_2, q_T, q_Td, eps, inc):
    def func(y):
        r, w = xyz_to_rw(x, y, z, inc)
        rho = (np.abs(r) + r_m - r_0) / r_m
        w_r = w_0 * rho ** eps

        if np.abs(r) < r_0:
            return 0.
        if w > w_r:
            return 0.

        p1 = T_0 * (1. + ((rho * (r_2 - r_1) * w) /
             (r_1 * w_0 * rho))) ** q_Td
        p2 = ((r + r_m - r_0) / r_m) ** q_T

        return p1 * p2
    return np.vectorize(func)

def kappa_y_func(x, z, n_0, x_0, T_0, w_0, r_m, r_0, r_1, r_2, q_n, q_nd, q_x,
                 q_xd, q_T, q_Td, eps, inc, freq):
    ne_y = ne_y_func(x, z, n_0, x_0, w_0, r_m, r_0, r_1, r_2, q_n, q_nd, q_x,
                     q_xd, eps, inc)
    T_y = T_y_func(x, z, T_0, w_0, r_m, r_0, r_1, r_2, q_T, q_Td, eps, inc)
    def func(y):
        r, w = xyz_to_rw(x, y, z, inc)
        rho = (np.abs(r) + r_m - r_0) / r_m
        w_r = w_0 * rho ** eps

        if np.abs(r) < r_0:
            return 0.
        if w > w_r:
            return 0.
        # p1 = 0.54e-38 * ne_y(y) ** 2. * (con.c * 1e2) ** 2.
        # p1 /= 2. * (con.k * 1e7) * T_y(y) ** 1.5 * freq ** 2.
        # p2 = np.exp(-(con.h * 1e7) * freq / ((con.k * 1e7) * T_y(y))) * \
        #      (11.95 * T_y(y) ** 0.15 * freq ** -0.1)
        #      #mphys.gff(freq, T_y(y), z=1.)
        # return p1 * p2
        return 0.212 * ne_y(y) ** 2. * T_y(y) ** -1.35 * freq ** -2.1

    return np.vectorize(func)

# g = Grid(nax, jm.params["grid"]["c_size"])
jm = JetModel(cfg.dcys['files'] + os.sep + 'example-model-params.py')
nz = jm.params["grid"]["n_z"]
nx = jm.params["grid"]["n_x"]
nsamples_y = 30
q_nd, q_xd, q_Td = -0., -0., -0.
freq = 10e9

ems = np.empty((nz, nx), dtype=float)
taus = np.empty((nz, nx), dtype=float)
# for idxr, row in enumerate(g.arr_centres[0,:,:,0].T):
#     for idxc, x in enumerate(row):
xs = np.linspace(jm.params["grid"]["c_size"] / 2.,
                 (nx - 0.5) * jm.params["grid"]["c_size"],
                 nx) - nx * jm.params["grid"]["c_size"] / 2
zs = np.linspace(jm.params["grid"]["c_size"] / 2.,
                 (nz - 0.5) * jm.params["grid"]["c_size"],
                 nz) - nz * jm.params["grid"]["c_size"] / 2


# zs = np.linspace(-10, 10, 1000)
# ys1, ys2 = y1_y2(0., zs, jm.params['geometry']['inc'],
#                  jm.params['geometry']['w_0'],
#                  jm.params['geometry']['mod_r_0'],
#                  jm.params['geometry']['r_0'])
#
#
# inc_rads = np.radians(jm.params["geometry"]["inc"])
#
# plt.close('all')
#
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
#
# # Plot y-intersection behind
# ax.plot(ys1, zs, 'r-')
#
# # Plot y-intersection in front
# ax.plot(ys2, zs, 'b-')
#
# # Plot central jet axis
# ax.plot(zs / np.tan(inc_rads), zs, 'k-')
#
# rs = zs / np.sin(inc_rads)
# rhos = (np.abs(rs) + jm.params['geometry']['mod_r_0'] -
#         jm.params['geometry']['r_0'])
# rhos /= jm.params['geometry']['mod_r_0']
# ws = jm.params['geometry']['w_0'] * rhos ** jm.params['geometry']['epsilon']
#
# w_r_ys1 = rs * np.cos(inc_rads) + ws * np.cos(np.pi / 2. - inc_rads)
# w_r_ys2 = rs * np.cos(inc_rads) - ws * np.cos(np.pi / 2. - inc_rads)
# w_r_zs1 = zs - ws * np.sin(np.pi / 2. - inc_rads)
# w_r_zs2 = zs + ws * np.sin(np.pi / 2. - inc_rads)
#
# ax.plot(w_r_ys1, w_r_zs1, 'b--')
# ax.plot(w_r_ys2, w_r_zs2, 'r--')
#
# ax_range = np.max([np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim())])
# ls = (0 - ax_range / 2., 0 + ax_range / 2.)
# ax.set_xlim(ls)
# ax.set_ylim(ls)
#
# plt.show()
#
ncells = nx * nz
count = 0
csize = jm.params["grid"]["c_size"]
for idxz, z in enumerate(zs):
    for idxx, x in enumerate(xs):
        count += 1

        # Position within 2D x/z-cell to integrate through in y
        xc = x - csize / 2 * np.sign(x)
        zc = z - csize / 2 * np.sign(z)

        ne_y = ne_y_func(xc, zc,
                         jm.params["properties"]["n_0"],
                         jm.params["properties"]["x_0"],
                         jm.params["geometry"]["w_0"],
                         jm.params["geometry"]["mod_r_0"],
                         jm.params["geometry"]["r_0"],
                         jm.params["target"]["r_1"],
                         jm.params["target"]["r_2"],
                         jm.params["power_laws"]["q_n"], q_nd,
                         jm.params["power_laws"]["q_x"], q_xd,
                         jm.params["geometry"]["epsilon"],
                         jm.params["geometry"]["inc"])

        k_y = kappa_y_func(xc, zc,
                           jm.params["properties"]["n_0"],
                           jm.params["properties"]["x_0"],
                           jm.params["properties"]["T_0"],
                           jm.params["geometry"]["w_0"],
                           jm.params["geometry"]["mod_r_0"],
                           jm.params["geometry"]["r_0"],
                           jm.params["target"]["r_1"],
                           jm.params["target"]["r_2"],
                           jm.params["power_laws"]["q_n"], q_nd,
                           jm.params["power_laws"]["q_x"], q_xd,
                           jm.params["power_laws"]["q_T"], q_Td,
                           jm.params["geometry"]["epsilon"],
                           jm.params["geometry"]["inc"], freq)
        y1, y2 = y1_y2(xc, zc,
                       jm.params["geometry"]["inc"],
                       jm.params["geometry"]["w_0"],
                       jm.params["geometry"]["mod_r_0"],
                       jm.params["geometry"]["r_0"])

        ys = np.linspace(y1, y2, nsamples_y)

        if True in np.isnan([y1, y2]):
            ems[idxz][idxx] = 0.
            taus[idxz][idxx] = 0.
        # ems[idxr][idxc] = quad(ne_y,
        #                        (ymin - g.cwidth / 2.) * 4.84814e-6,
        #                        (ymax + g.cwidth / 2.) * 4.84814e-6)[0]
        # intgrl = cumtrapz(ne_y(ys) ** 2., ys * con.au / con.parsec)
        # intgrl2 = trapz(ne_y(ys) ** 2., ys * con.au / con.parsec)
        else:
            ems[idxz][idxx] = trapz(ne_y(ys) ** 2., ys * con.au /
                                       con.parsec)
            taus[idxz][idxx] = trapz(k_y(ys), ys * con.au * 1e2)
            # taus[idxz][idxx] = quad(k_y, y1 * con.au * 1e2,
            #                         y2 * con.au * 1e2)[0]

        print('{:3.0f}% complete'.format(count / ncells * 100),
              end='\r' if count != ncells else '\n')

t_bs = jm.params["properties"]["T_0"] * (1. - np.exp(-taus))
i_nus = t_bs * 2. * con.k * freq**2. / con.c ** 2.

d_omega = np.tan(jm.params["grid"]["c_size"] * con.au / con.parsec /
                 jm.params["target"]["dist"])**2.
fluxes = i_nus * d_omega
fluxes /= 1e-26 * 1e-6  # W m^-2 Hz^-1 to uJy

print("""
{:.2f} uJy from model
{:.2f} uJy from R86
""".format(np.nansum(fluxes),
           mphys.flux_expected_r86(jm, freq,
                                   np.max(zs) / jm.params['target']['dist']) * 2 * 1e6))

# f = func1(jm.params["grid"]["c_size"],
#           jm.params["properties"]["n_0"],
#           jm.params["geometry"]["w_0"], jm.params["geometry"]["mod_r_0"],
#           jm.params["geometry"]["r_0"], jm.params["target"]["r_1"],
#           jm.params["target"]["r_2"], jm.params["power_laws"]["q_n"],
#           q_nd, jm.params["geometry"]["epsilon"],
#           jm.params["geometry"]["inc"])

import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
plt.close('all')

plt.imshow(np.where(ems == 0, np.NaN, ems),
           extent=(np.min(xs) - jm.params["grid"]["c_size"] / 2.,
                   np.max(xs) + jm.params["grid"]["c_size"] / 2.,
                   np.min(zs) - jm.params["grid"]["c_size"] / 2.,
                   np.max(zs) + jm.params["grid"]["c_size"] / 2.),
           norm=LogNorm(vmin=np.nanmin(np.where(ems == 0, np.NaN, ems)),
                        vmax=np.nanmax(np.where(ems == 0, np.NaN, ems))))

plt.imshow(np.where(taus == 0, np.NaN, taus),
           extent=(np.min(xs) - jm.params["grid"]["c_size"] / 2.,
                   np.max(xs) + jm.params["grid"]["c_size"] / 2.,
                   np.min(zs) - jm.params["grid"]["c_size"] / 2.,
                   np.max(zs) + jm.params["grid"]["c_size"] / 2.),
           norm=LogNorm(vmin=np.nanmin(np.where(taus == 0, np.NaN, taus)),
                        vmax=np.nanmax(np.where(taus == 0, np.NaN, taus))))

plt.imshow(np.where(fluxes == 0, np.NaN, fluxes),
           extent=(np.min(xs) - jm.params["grid"]["c_size"] / 2.,
                   np.max(xs) + jm.params["grid"]["c_size"] / 2.,
                   np.min(zs) - jm.params["grid"]["c_size"] / 2.,
                   np.max(zs) + jm.params["grid"]["c_size"] / 2.),
           norm=LogNorm(vmin=np.nanmin(np.where(fluxes == 0, np.NaN, fluxes)),
                        vmax=np.nanmax(np.where(fluxes == 0, np.NaN, fluxes))))

ws =  jm.params["geometry"]["w_0"] * \
      ((zs / np.sin(np.radians(jm.params["geometry"]["inc"])) +
        jm.params["geometry"]["mod_r_0"] - jm.params["geometry"]["r_0"]) /
       jm.params["geometry"]["mod_r_0"]) **  jm.params["geometry"]["epsilon"]

ws = np.where(zs < jm.params["geometry"]["r_0"] *
              np.sin(np.radians(jm.params["geometry"]["inc"])), np.NaN, ws)
plt.plot(ws, zs, 'r-')
plt.plot(ws, -zs, 'r-')
plt.plot(-ws, zs, 'r-')
plt.plot(-ws, -zs, 'r-')

plt.show()

# ns = np.empty_like(g.arr_cells, dtype=float)
# count = 0.
# ncells = np.prod(np.shape(g.arr_cells))
# for idxx, xplane in enumerate(g.arr_cells):
#     for idxy, ycol in enumerate(xplane):
#         for idxz, cell in enumerate(ycol):
#             x, y, z = cell.centre
#             ns[idxx][idxy][idxz] = n_xyz(f, x, y, z,
#                                          jm.params["grid"]["c_size"])
#             count += 1.
#             print('{:3.0f}% complete'.format(count / ncells * 100.), end='\r')
