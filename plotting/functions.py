# -*- coding: utf-8 -*-
from typing import Union
import numpy as np
import scipy.constants as con
import matplotlib.axes
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm, SymLogNorm, Normalize
from matplotlib.ticker import AutoLocator, AutoMinorLocator, FuncFormatter
from matplotlib.ticker import MultipleLocator, MaxNLocator
import astropy.units as u
# from RaJePy import cnsts
# from RaJePy import JetModel
# from RaJePy import _config as cfg
# from RaJePy.maths import physics as mphys


def equalise_axes(ax, fix_x=False, fix_y=False, fix_z=False):
    """
    Equalises the x/y/z axes of a matplotlib.axes._subplots.AxesSubplot
    instance. Autodetects if 2-D, 3-D, linear-scaling or logarithmic-scaling.
    fix_x/fix_y/fix_z is to fix the x, y or z scales of the plot and work
    the other axes limits around it, potentially chopping off data points. Only
    one of fix_x/fix_y/fix_z can be true
    """
    if sum((fix_x, fix_y, fix_z)) not in (0, 1):
        raise ValueError("Only 1 of fix_x, fix_y or fix_z can be set to True "
                         "as a maximum")

    if ax.get_xscale() == 'log':
        logx = True
    else:
        logx = False
    if ax.get_yscale() == 'log':
        logy = True
    else:
        logy = False
    try:
        if ax.get_zscale():
            logz = True
        else:
            logz = False
        ndims = 3
    except AttributeError:
        ndims = 2
        logz = False

    x_range = np.ptp(ax.get_xlim())
    y_range = np.ptp(ax.get_ylim())
    if ndims == 3:
        z_range = np.ptp(ax.get_zlim())
    else:
        z_range = None

    if logx:
        x_range = np.ptp(np.log10(ax.get_xlim()))
    if logy:
        y_range = np.ptp(np.log10(ax.get_ylim()))
    if ndims == 3 and logz:
        z_range = np.ptp(np.log10(ax.get_zlim()))

    if ndims == 3:
        r = np.max([x_range, y_range, z_range])
    else:
        r = np.max([x_range, y_range])

    if fix_x:
        r = x_range
    elif fix_y:
        r = y_range
    elif ndims == 3 and fix_z:
        r = z_range

    if logx:
        xlims = (10 ** (np.mean(np.log10(ax.get_xlim())) - r / 2.),
                 10 ** (np.mean(np.log10(ax.get_xlim())) + r / 2.))
    else:
        xlims = (np.mean(ax.get_xlim()) - r / 2.,
                 np.mean(ax.get_xlim()) + r / 2.)
    ax.set_xlim(xlims)

    if logy:
        ylims = (10 ** (np.mean(np.log10(ax.get_ylim())) - r / 2.),
                 10 ** (np.mean(np.log10(ax.get_ylim())) + r / 2.))
    else:
        ylims = (np.mean(ax.get_ylim()) - r / 2.,
                 np.mean(ax.get_ylim()) + r / 2.)
    ax.set_ylim(ylims)

    if ndims == 3:
        if logz:
            zlims = (10 ** (np.mean(np.log10(ax.get_zlim())) - r / 2.),
                     10 ** (np.mean(np.log10(ax.get_zlim())) + r / 2.))
        else:
            zlims = (np.mean(ax.get_zlim()) - r / 2.,
                     np.mean(ax.get_zlim()) + r / 2.)
        ax.set_zlim(zlims)

        return xlims, ylims, zlims

    return xlims, ylims


def make_colorbar(cax, cmax, cmin=0, position='right', orientation='vertical',
                  numlevels=50, colmap='viridis', norm=None,
                  maxticks=AutoLocator(), minticks=False, tickformat=None,
                  hidespines=False):
    # Custom colorbar using axes so that can set colorbar properties straightforwardly

    if isinstance(norm, LogNorm):
        colbar = np.linspace(np.log10(cmin), np.log10(cmax), numlevels + 1)
    elif isinstance(norm, SymLogNorm):
        raise NotImplementedError
    else:
        colbar = np.linspace(cmin, cmax, numlevels + 1)

    levs = []
    for e, E in enumerate(colbar, 0):
        if e < len(colbar) - 1:
            if isinstance(norm, LogNorm):
                levs = np.concatenate((levs[:-1], np.linspace(10 ** colbar[e],
                                                              10 ** colbar[e + 1],
                                                              numlevels)))
            else:
                levs = np.concatenate((levs[:-1], np.linspace(colbar[e],
                                                              colbar[e + 1],
                                                              numlevels)))
    yc = [levs, levs]
    xc = [np.zeros(len(levs)), np.ones(len(levs))]

    if np.ptp(levs) == 0:
        if isinstance(norm, LogNorm):
            levs = np.logspace(np.log10(levs[0]) - 1, np.log10(levs[0]),
                               len(xc[0]))
        else:
            levs = np.linspace(levs[0] * 0.1, levs[0], len(xc[0]))

    if orientation == 'vertical':
        cax.contourf(xc, yc, yc, cmap=colmap, levels=levs, norm=norm)
        cax.yaxis.set_ticks_position(position)
        cax.xaxis.set_ticks([])
        axis = cax.yaxis
    elif orientation == 'horizontal':
        cax.contourf(yc, xc, yc, cmap=colmap, levels=levs, norm=norm)
        cax.xaxis.set_ticks_position(position)
        cax.yaxis.set_ticks([])
        axis = cax.xaxis
    else:
        raise ValueError("Orientation must be 'vertical' or 'horizontal'")

    if isinstance(norm, LogNorm):
        if orientation == 'vertical':
            cax.set_yscale('log')  # , subsy=minticks if isinstance(minticks, list) else [1, 2, 3, 4, 5, 6, 7, 8, 9])
        elif orientation == 'horizontal':
            cax.set_xscale('log')  # , subsy=minticks if isinstance(minticks, list) else [1, 2, 3, 4, 5, 6, 7, 8, 9])
    else:
        if isinstance(maxticks, list):
            axis.set_ticks(maxticks)
        elif isinstance(maxticks, (AutoLocator, AutoMinorLocator, MultipleLocator, MaxNLocator)):
            axis.set_major_locator(maxticks)

        if isinstance(minticks, list):
            axis.set_ticks(minticks, minor=True)
        elif isinstance(minticks, (AutoLocator, AutoMinorLocator, MultipleLocator, MaxNLocator)):
            axis.set_minor_locator(minticks)
        elif minticks:
            axis.set_minor_locator(AutoMinorLocator())

    if tickformat:
        if orientation == 'vertical':
            cax.yaxis.set_major_formatter(FuncFormatter(tickformat))
        elif orientation == 'horizontal':
            cax.xaxis.set_major_formatter(FuncFormatter(tickformat))

    if hidespines:
        for spine in ['left', 'bottom', 'top']:
            cax.spines[spine].set_visible(False)


def plot_mass_volume_slices(jm: 'JetModel', show_plot: bool = False,
                            savefig: Union[bool, str] = False):
    """
    Plot mass and volume slices as check for consistency (i.e. mass is
    are conserved). Only really makes sense for jet models with  i = 90deg and
    pa = 0deg.

    Parameters
    ----------
    jm
        JetModel instance from which to plot mass/volume slices.
    show_plot
        Whether to show the plot on the display device. Useful for interactive
        console sessions, False by default
    savefig
        Whether to save the figure, False by default. Provide the full path of
        the save file as a str to save.

    Returns
    -------
    None
    """
    from RaJePy import cnsts
    from RaJePy.maths import physics as mphys
    from RaJePy import _config as cfg

    def m_slice(_a, _b):
        """
        Mass of slice over the interval from a --> b in z,
        in kg calculated from model parameters
        """
        n_0 = jm.params["properties"]["n_0"] * 1e6
        mod_r_0 = jm.params["geometry"]["mod_r_0"] * con.au
        r_0 = jm.params["geometry"]["r_0"] * con.au
        q_n = jm.params["power_laws"]["q_n"]
        w_0 = jm.params["geometry"]["w_0"] * con.au
        eps = jm.params["geometry"]["epsilon"]
        mu = jm.params['properties']['mu'] * mphys.atomic_mass("H")

        def indef_integral(z):
            """Volume of slice over the interval from a --> b in z,
            in m^3 calculated from model parameters"""
            c = 1 + q_n + 2. * eps
            num_p1 = mu * np.pi * mod_r_0 * n_0 * w_0 ** 2.
            num_p2 = ((z + mod_r_0 - r_0) / mod_r_0) ** c

            return num_p1 * num_p2 / c

        return indef_integral(_b) - indef_integral(_a)

    def v_slice(_a, _b):
        """
        Volume of slice over the interval from a --> b in z, in m^3
        """
        mod_r_0 = jm.params["geometry"]["mod_r_0"] * con.au
        r_0 = jm.params["geometry"]["r_0"] * con.au
        w_0 = jm.params["geometry"]["w_0"] * con.au
        eps = jm.params["geometry"]["epsilon"]

        def indef_integral(z):
            c = 1 + 2. * eps
            num_p1 = np.pi * mod_r_0 * w_0 ** 2.
            num_p2 = ((z + mod_r_0 - r_0) / mod_r_0) ** c

            return num_p1 * num_p2 / c

        return indef_integral(_b) - indef_integral(_a)

    a = np.abs(jm.zs + jm.csize / 2) - jm.csize / 2
    b = np.abs(jm.zs + jm.csize / 2) + jm.csize / 2

    a = np.where(b <= jm.params['geometry']['r_0'], np.NaN, a)
    b = np.where(b <= jm.params['geometry']['r_0'], np.NaN, b)
    a = np.where(a <= jm.params['geometry']['r_0'],
                 jm.params['geometry']['r_0'], a)

    a *= con.au
    b *= con.au

    # Use the above functions to calculate what each slice's mass should be
    mslices_calc = m_slice(a, b)
    vslices_calc = v_slice(a, b)

    # Calculate cell volumes and slice volumes
    vcells = jm.fill_factor * (jm.csize * con.au) ** 3.
    vslices = np.nansum(np.nansum(vcells, axis=1), axis=1)

    # Calculate mass density of cells (in kg m^-3)
    mdcells = jm.number_density * jm.params['properties']['mu'] * mphys.atomic_mass("H") * 1e6

    # Calculate cell masses
    mcells = mdcells * vcells

    # Sum cell masses to get slice masses
    mslices = np.nansum(np.nansum(mcells, axis=1), axis=1)

    vslices_calc /= con.au ** 3.
    mslices_calc /= cnsts.MSOL
    vslices /= con.au ** 3.
    mslices /= cnsts.MSOL

    # verrs = vslices - vslices_calc
    # merrs = mslices - mslices_calc

    vratios = vslices / vslices_calc
    mratios = mslices / mslices_calc

    # Average z-value for each slice
    zs = np.mean([a, b], axis=1) / con.au
    zs *= np.sign(jm.zs + jm.csize / 2)

    plt.close('all')

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                                   figsize=[cfg.plots["dims"]["column"],
                                            cfg.plots["dims"][
                                                "column"] * 2])

    ax1b = ax1.twinx()
    ax2b = ax2.twinx()
    ax1b.sharey(ax2b)

    ax1b.plot(zs, vratios, ls='-', zorder=1, color='slategrey', lw=2)
    ax2b.plot(zs, mratios, ls='-', zorder=1, color='slategrey', lw=2)

    for ax in (ax1b, ax2b):
        ax.tick_params(which='both', direction='in', color='slategrey')
        ax.tick_params(axis='y', which='both', colors='slategrey')
        ax.spines['right'].set_color('slategrey')
        ax.yaxis.label.set_color('slategrey')
        ax.minorticks_on()

    ax1.tick_params(axis='x', labelbottom=False)

    ax1.plot(zs, vslices, color='r', ls='-', zorder=2)
    ax1.plot(zs, vslices_calc, color='b', ls=':', zorder=3)

    ax2.plot(zs, mslices, color='r', ls='-', zorder=2)
    ax2.plot(zs, mslices_calc, color='b', ls=':', zorder=3)

    ax2.set_xlabel(r"$z \, \left[ {\rm au} \right]$")

    ax1.set_ylabel(r"$V_{\rm slice} \, \left[ {\rm au}^3 \right]$")
    ax2.set_ylabel(r"$M_{\rm slice} \, \left[ {\rm M}_\odot \right]$")

    ax1.tick_params(which='both', direction='in', top=True)
    ax2.tick_params(which='both', direction='in', top=True)

    ax1b.set_ylabel(r"$\ \frac{V^{\rm model}_{\rm slice}}"
                    r"{V^{\rm actual}_{\rm slice}}$")
    ax2b.set_ylabel(r"$\ \frac{M^{\rm model}_{\rm slice}}"
                    r"{M^{\rm actual}_{\rm slice}}$")

    plt.subplots_adjust(wspace=0, hspace=0)

    ax1b.set_ylim(0, 1.99)

    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)

    ax1.set_zorder(ax1b.get_zorder() + 1)
    ax2.set_zorder(ax2b.get_zorder() + 1)

    # Set ax's patch invisible
    ax1.patch.set_visible(False)
    ax2.patch.set_visible(False)

    # Set axtwin's patch visible and colorize its background white
    ax1b.patch.set_visible(True)
    ax2b.patch.set_visible(True)
    ax1b.patch.set_facecolor('white')
    ax2b.patch.set_facecolor('white')

    if savefig:
        # TODO: Put this in appropriate place in JetModel class
        # jm.log.add_entry("INFO",
        #                    "Diagnostic plot saved to " + savefig)
        plt.savefig(savefig, bbox_inches='tight', dpi=300)

    if show_plot:
        plt.show()

    return None


def diagnostic_plot(jm: 'JetModel', show_plot: bool = False,
                    savefig: Union[bool, str] = False):
    """
    Plots mass/angular momentum slices as functions of distance along the jet to
    check for conservation of both quantities.

    Parameters
    ----------
    jm
        JetModel instance from which to plot mass/volume slices.
    show_plot
        Whether to show the plot on the display device. Useful for interactive
        console sessions, False by default
    savefig
        Whether to save the figure, False by default. Provide the full path of
        the save file as a str to save.

    Returns
    -------
    None
    """
    from RaJePy import _config as cfg

    inc = jm.params['geometry']['inc']
    pa = jm.params['geometry']['pa']
    if inc != 90. or pa != 0.:
        # TODO: Put this in appropriate place in JetModel class
        # jm.log.add_entry("WARNING",
        #                    "Diagnostic plots may be increasingly "
        #                    "inaccurate for inclined or rotated jets"
        #                    " (i.e. i != 90 deg or pa != 0 deg")
        return None

    # Conservation of mass, angular momentum and energy
    # cell_vol = (jm.csize * con.au * 1e2) ** 3.
    # particle_mass = jm.params['properties']['mu'] * con.u
    masses = jm.mass
    vxs, vys = jm.vel[:2]

    # TODO: Method needed to calculate v_w whilst incorporating the
    #  effects of inclination and position angle
    vws = np.sqrt(vxs ** 2. + vys ** 2.)
    angmoms = masses * (vws * 1000.) * (jm.ww * con.au)

    if inc == 90. and pa == 0.:
        masses_slices = np.nansum(masses, axis=(0, 1))
        angmom_slices = np.nansum(angmoms, axis=(0, 1))
        rs = jm.rr[0][0]
    # Following not implemented yet as vws need to be accurately calculated
    else:
        rs = np.arange(jm.csize / 2., np.nanmax(jm.rr), jm.csize)
        rs = np.append(-rs, rs)
        masses_slices = []
        angmom_slices = []
        for r in rs:
            mask = ((jm.rr >= (r - jm.csize / 2.)) &
                    (jm.rr <= (r + jm.csize / 2.)))
            masses_slices.append(np.nansum(np.where(mask, masses, np.NaN)))
            angmom_slices.append(np.nansum(np.where(mask, angmoms, np.NaN)))

    plt.close('all')

    fig, (ax1, ax2) = plt.subplots(2, 1,
                                   figsize=(cfg.plots['dims']['column'],
                                            cfg.plots['dims']['text']),
                                   sharex=True)

    ax1.plot(rs, masses_slices, 'b-')
    ax2.plot(rs, angmom_slices, 'r-')

    ax2.set_xlabel(r'$r\,\left[\mathrm{au}\right]$')

    ax1.set_ylabel(r'$m\,\left[\mathrm{kg}\right]$')
    ax2.set_ylabel(r'$L\,\left[\mathrm{kg\,m^2\,s{-1}}\right]$')

    for ax in (ax1, ax2):
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.minorticks_on()

    plt.subplots_adjust(wspace=0, hspace=0)

    if savefig:
        # TODO: Put this in appropriate place in JetModel class
        # jm.log.add_entry("INFO",
        #                    "Diagnostic plot saved to " + savefig)
        plt.savefig(savefig, bbox_inches='tight', dpi=300)

    if show_plot:
        plt.show()

    return None


def model_plot(jm: 'JetModel', show_plot: bool = False,
               savefig: Union[bool, str] = False):
    """
    Generate 4 subplots of (from top left, clockwise) number density,
    temperature, ionisation fraction and velocity.

    Parameters
    ----------
    jm
        JetModel instance from which to plot mass/volume slices.
    show_plot
        Whether to show the plot on the display device. Useful for interactive
        console sessions, False by default
    savefig
        Whether to save the figure, False by default. Provide the full path of
        the save file as a str to save.

    Returns
    -------
    None

    """
    from RaJePy import _config as cfg

    plt.close('all')

    fig = plt.figure(figsize=([cfg.plots["dims"]["column"] * 2.] * 2))

    # Set common labels
    fig.text(0.5, 0.025, r'$\Delta x \, \left[ {\rm au} \right]$',
             ha='center', va='bottom')
    fig.text(0.025, 0.5, r'$\Delta z \, \left[ {\rm au} \right] $',
             ha='left', va='center', rotation='vertical')

    outer_grid = gridspec.GridSpec(2, 2)

    tl_cell = gridspec.GridSpecFromSubplotSpec(1, 2, outer_grid[0, 0],
                                               width_ratios=[9, 1],
                                               wspace=0.0, hspace=0.0)

    # Number density
    tl_ax = plt.subplot(tl_cell[0, 0])
    tl_cax = plt.subplot(tl_cell[0, 1])

    tr_cell = gridspec.GridSpecFromSubplotSpec(1, 2, outer_grid[0, 1],
                                               width_ratios=[9, 1],
                                               wspace=0.0, hspace=0.0)

    # Temperature
    tr_ax = plt.subplot(tr_cell[0, 0])
    tr_cax = plt.subplot(tr_cell[0, 1])

    bl_cell = gridspec.GridSpecFromSubplotSpec(1, 2, outer_grid[1, 0],
                                               width_ratios=[9, 1],
                                               wspace=0.0, hspace=0.0)

    # Ionisation fraction
    bl_ax = plt.subplot(bl_cell[0, 0])
    bl_cax = plt.subplot(bl_cell[0, 1])

    br_cell = gridspec.GridSpecFromSubplotSpec(1, 2, outer_grid[1, 1],
                                               width_ratios=[9, 1],
                                               wspace=0.0, hspace=0.0)

    # Velocity los-component
    br_ax = plt.subplot(br_cell[0, 0])
    br_cax = plt.subplot(br_cell[0, 1])

    axes = tl_ax, tr_ax, bl_ax, br_ax
    caxes = tl_cax, tr_cax, bl_cax, br_cax

    bbox = tl_ax.get_window_extent()
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    aspect = bbox.width / bbox.height
    im_extent = (np.min(jm.xx), np.max(jm.xx) + jm.csize * 1.,
                 np.min(jm.zz), np.max(jm.zz) + jm.csize * 1.)

    # Number densities in top-left plot
    ns = np.nanmean(jm.number_density, axis=jm.los_axis)
    im_nd = tl_ax.imshow(np.swapaxes(ns, 0, 1),
                         norm=LogNorm(vmin=np.nanmin(jm.number_density),
                                      vmax=np.nanmax(jm.number_density)),
                         extent=im_extent,
                         cmap='viridis_r', aspect="equal", origin="lower")
    tl_ax.set_xlim(np.array(tl_ax.get_ylim()) * aspect)
    make_colorbar(tl_cax, np.nanmax(jm.number_density),
                  cmin=np.nanmin(jm.number_density),
                  position='right', orientation='vertical',
                  numlevels=50, colmap='viridis_r', norm=im_nd.norm)

    # Temperatures in top-right plot
    temps = np.nanmean(jm.temperature, axis=jm.los_axis)
    im_T = tr_ax.imshow(np.swapaxes(temps, 0, 1),
                        norm=LogNorm(vmin=100.,
                                     vmax=max([1e4, np.nanmax(
                                         jm.temperature)])),
                        extent=im_extent,
                        cmap='plasma', aspect="equal", origin="lower")
    tr_ax.set_xlim(np.array(tr_ax.get_ylim()) * aspect)
    make_colorbar(tr_cax, max([1e4, np.nanmax(jm.temperature)]),
                  cmin=100., position='right',
                  orientation='vertical', numlevels=50,
                  colmap='plasma', norm=im_T.norm)
    tr_cax.set_ylim(100., 1e4)

    # Ionisation fractions in bottom-left plot
    xis = np.nanmean(jm.ion_fraction, axis=jm.los_axis) * 100.
    im_xi = bl_ax.imshow(np.swapaxes(xis, 0, 1),
                         vmin=0., vmax=100.0,
                         extent=im_extent,
                         cmap='gnuplot', aspect="equal", origin="lower")
    bl_ax.set_xlim(np.array(bl_ax.get_ylim()) * aspect)
    make_colorbar(bl_cax, 100., cmin=0., position='right',
                  orientation='vertical', numlevels=50,
                  colmap='gnuplot', norm=im_xi.norm)
    bl_cax.set_yticks(np.linspace(0., 100., 6))

    # Line-of-sight velocities (corrected for V_lsr) in bottom-right plot
    v_los = np.nanmean(jm.vel[1] - jm.params["target"]["v_lsr"],
                       axis=jm.los_axis)
    lim_v_los = np.nanmax(np.abs(v_los))
    im_vs = br_ax.imshow(np.swapaxes(v_los, 0, 1),
                         vmin=-lim_v_los, vmax=lim_v_los,
                         extent=im_extent,
                         cmap='coolwarm', aspect="equal", origin="lower")
    br_ax.set_xlim(np.array(br_ax.get_ylim()) * aspect)
    make_colorbar(br_cax, lim_v_los, cmin=-lim_v_los, position='right',
                  orientation='vertical', numlevels=50,
                  colmap='coolwarm', norm=im_vs.norm)

    # dx = int((np.ptp(br_ax.get_xlim()) / jm.csize) // 2 * 2 // 20)
    # dz = jm.nz // 10
    #
    # vxs = jm.vel[0][::dx, jm.ny // 2, ::dz]
    # vxs = np.flip(np.swapaxes(vxs, 0, 1), axis=0)
    #
    # vzs = jm.vel[2][::dx, jm.ny // 2, ::dz]
    # vzs = np.flip(np.swapaxes(vzs, 0, 1), axis=0)
    #
    # xs = jm.xx[::dx, jm.ny // 2, ::dz]
    # xs = np.flip(np.swapaxes(xs, 0, 1), axis=0)#[~np.isnan(vzs)]
    #
    # zs = jm.zz[::dx, jm.ny // 2, ::dz]
    # zs = np.flip(np.swapaxes(zs, 0, 1), axis=0)#[~np.isnan(vzs)]
    #
    # vxs = vxs[~np.isnan(vxs)]
    # vzs = vzs[~np.isnan(vzs)]
    #
    # max_vs_pos = np.nanmax(np.sqrt(vxs ** 2. + vzs ** 2.))
    #
    # cs = br_ax.transAxes.transform((0.15, 0.5))
    # cs = br_ax.transData.inverted().transform(cs)
    #
    # # TODO: This is broken. vzs are inverted in this plotting routine. They are ok in the model though...
    # try:
    #     v_scale = np.ceil(max_vs_pos / 10 ** np.floor(np.log10(max_vs_pos)))
    #     v_scale *= 10 ** np.floor(np.log10(max_vs_pos))
    #
    #     # Max arrow length is 0.1 * the height of the subplot
    #     scale = v_scale * 0.1 ** -1.
    #     br_ax.quiver(xs.flatten(), zs.flatten(), vxs.flatten(), vzs.flatten(),
    #                  color='w', scale=scale,
    #                  scale_units='height')
    #
    #     br_ax.quiver(cs[0], cs[1], [0.], [v_scale], color='k', scale=scale,
    #                  scale_units='height', pivot='tail')
    #
    #     br_ax.annotate(r'$' + format(v_scale, '.0f') + '$\n$' +
    #                    r'\rm{km/s}$', cs, xytext=(0., -5.),  # half fontsize
    #                    xycoords='data', textcoords='offset points',
    #                    va='top',
    #                    ha='center', multialignment='center', fontsize=10)
    # except ValueError:
    #     pass

    tl_ax.text(0.9, 0.9, r'a', ha='center', va='center',
               transform=tl_ax.transAxes)
    tr_ax.text(0.9, 0.9, r'b', ha='center', va='center',
               transform=tr_ax.transAxes)
    bl_ax.text(0.9, 0.9, r'c', ha='center', va='center',
               transform=bl_ax.transAxes)
    br_ax.text(0.9, 0.9, r'd', ha='center', va='center',
               transform=br_ax.transAxes)

    tl_ax.axes.xaxis.set_ticklabels([])
    tr_ax.axes.xaxis.set_ticklabels([])
    tr_ax.axes.yaxis.set_ticklabels([])
    br_ax.axes.yaxis.set_ticklabels([])

    for ax in axes:
        ax.set_xlim(np.nanmin(im_extent), np.nanmax(im_extent))
        ax.set_ylim(np.nanmin(im_extent), np.nanmax(im_extent))

    for ax in axes:
        if np.ptp(ax.get_xlim()) > np.ptp(ax.get_ylim()):
            xlims = ax.get_ylim()
            ax.set_xticks(ax.get_yticks())
            ax.set_xlim(xlims)
        else:
            ylims = ax.get_xlim()
            ax.set_yticks(ax.get_xticks())
            ax.set_ylim(ylims)
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.minorticks_on()

    tl_cax.text(0.5, 0.5, r'$\left[{\rm cm^{-3}}\right]$', ha='center',
                va='center', transform=tl_cax.transAxes, color='white',
                rotation=90.)
    tr_cax.text(0.5, 0.5, r'$\left[{\rm K}\right]$', ha='center',
                va='center', transform=tr_cax.transAxes, color='white',
                rotation=90.)
    bl_cax.text(0.5, 0.5, r'$\left[\%\right]$', ha='center', va='center',
                transform=bl_cax.transAxes, color='white', rotation=90.)
    br_cax.text(0.5, 0.5, r'$\left[{\rm km\,s^{-1}}\right]$', ha='center',
                va='center', transform=br_cax.transAxes, color='white',
                rotation=90.)

    for cax in caxes:
        cax.yaxis.set_label_position("right")
        cax.minorticks_on()

    if savefig:
        # TODO: Put this log entry in appropriate place in JetModel class
        # jm.log.add_entry("INFO",
        #                    "Model plot saved to " + savefig)
        plt.savefig(savefig, bbox_inches='tight', dpi=300)
        plt.close('all')

    if show_plot:
        plt.show()

    return None


def rt_plot(jm: 'JetModel', freq: float, percentile: float = 5.,
            show_plot: bool = False, savefig: Union[bool, str] = False):
    """
    Generate the 3 subplots of radiative transfer solutions (from left to right)
    flux, optical depth and emission measure.

    Parameters
    ----------
    jm
        JetModel instance from which to plot mass/volume slices.
    freq
        Frequency to produce images at.
    percentile
        Percentile of pixels to exclude from colorscale. Implemented as
        some edge pixels have extremely low values. Supplied value must be
        between 0 and 100.
    savefig: bool, str
        Whether to save the radio plot to file. If False, will not, but if
        a str representing a valid path will save to that path.
    show_plot
        Whether to show the plot on the display device. Useful for interactive
        console sessions, False by default
    savefig
        Whether to save the figure, False by default. Provide the full path of
        the save file as a str to save.

    Returns
    -------
    None
    """

    plt.close('all')

    fig = plt.figure(figsize=(6.65, 6.65 / 2))

    # Set common labels
    fig.text(0.5, 0.0, r'$\Delta\alpha\,\left[^{\prime\prime}\right]$',
             ha='center', va='bottom')
    fig.text(0.05, 0.5, r'$\Delta\delta\,\left[^{\prime\prime}\right]$',
             ha='left', va='center', rotation='vertical')

    outer_grid = gridspec.GridSpec(1, 3, wspace=0.4)

    # Flux
    l_cell = gridspec.GridSpecFromSubplotSpec(1, 2, outer_grid[0, 0],
                                              width_ratios=[5.667, 1],
                                              wspace=0.0, hspace=0.0)
    l_ax = plt.subplot(l_cell[0, 0])
    l_cax = plt.subplot(l_cell[0, 1])

    # Optical depth
    m_cell = gridspec.GridSpecFromSubplotSpec(1, 2, outer_grid[0, 1],
                                              width_ratios=[5.667, 1],
                                              wspace=0.0, hspace=0.0)
    m_ax = plt.subplot(m_cell[0, 0])
    m_cax = plt.subplot(m_cell[0, 1])

    # Emission measure
    r_cell = gridspec.GridSpecFromSubplotSpec(1, 2, outer_grid[0, 2],
                                              width_ratios=[5.667, 1],
                                              wspace=0.0, hspace=0.0)
    r_ax = plt.subplot(r_cell[0, 0])
    r_cax = plt.subplot(r_cell[0, 1])

    bbox = l_ax.get_window_extent()
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    aspect = bbox.width / bbox.height

    flux = jm.flux_ff(freq) * 1e3
    taus = jm.optical_depth_ff(freq)
    taus = np.where(taus > 0, taus, np.NaN)
    ems = jm.emission_measure()
    ems = np.where(ems > 0., ems, np.NaN)

    csize_as = np.tan(jm.csize * con.au / con.parsec /
                      jm.params['target']['dist'])  # radians
    csize_as /= con.arcsec  # arcseconds
    x_extent = np.shape(flux)[0] * csize_as
    z_extent = np.shape(flux)[1] * csize_as

    flux_min = np.nanpercentile(flux, percentile)
    im_flux = l_ax.imshow(flux.T,
                          norm=LogNorm(vmin=flux_min,
                                       vmax=np.nanmax(flux)),
                          extent=(-x_extent / 2., x_extent / 2.,
                                  -z_extent / 2., z_extent / 2.),
                          cmap='gnuplot2_r', aspect="equal")

    l_ax.set_xlim(np.array(l_ax.get_ylim()) * aspect)
    make_colorbar(l_cax, np.nanmax(flux), cmin=flux_min,
                  position='right', orientation='vertical',
                  numlevels=50, colmap='gnuplot2_r',
                  norm=im_flux.norm)

    tau_min = np.nanpercentile(taus, percentile)
    im_tau = m_ax.imshow(taus.T,
                         norm=LogNorm(vmin=tau_min,
                                      vmax=np.nanmax(taus)),
                         extent=(-x_extent / 2., x_extent / 2.,
                                 -z_extent / 2., z_extent / 2.),
                         cmap='Blues', aspect="equal")
    m_ax.set_xlim(np.array(m_ax.get_ylim()) * aspect)
    make_colorbar(m_cax, np.nanmax(taus), cmin=tau_min,
                  position='right', orientation='vertical',
                  numlevels=50, colmap='Blues',
                  norm=im_tau.norm)

    em_min = np.nanpercentile(ems, percentile)
    im_EM = r_ax.imshow(ems.T,
                        norm=LogNorm(vmin=em_min,
                                     vmax=np.nanmax(ems)),
                        extent=(-x_extent / 2., x_extent / 2.,
                                -z_extent / 2., z_extent / 2.),
                        cmap='cividis', aspect="equal")
    r_ax.set_xlim(np.array(r_ax.get_ylim()) * aspect)
    make_colorbar(r_cax, np.nanmax(ems), cmin=em_min,
                  position='right', orientation='vertical',
                  numlevels=50, colmap='cividis',
                  norm=im_EM.norm)

    axes = [l_ax, m_ax, r_ax]
    caxes = [l_cax, m_cax, r_cax]

    l_ax.text(0.9, 0.9, r'a', ha='center', va='center',
              transform=l_ax.transAxes)
    m_ax.text(0.9, 0.9, r'b', ha='center', va='center',
              transform=m_ax.transAxes)
    r_ax.text(0.9, 0.9, r'c', ha='center', va='center',
              transform=r_ax.transAxes)

    m_ax.axes.yaxis.set_ticklabels([])
    r_ax.axes.yaxis.set_ticklabels([])

    for ax in axes:
        ax.contour(np.linspace(-x_extent / 2., x_extent / 2.,
                               np.shape(flux)[0]),
                   np.linspace(-z_extent / 2., z_extent / 2.,
                               np.shape(flux)[1]),
                   taus.T, [1.], colors='w')
        xlims = ax.get_xlim()
        ax.set_xticks(ax.get_yticks())
        ax.set_xlim(xlims)
        ax.tick_params(which='both', direction='in', top=True,
                       right=True)
        ax.minorticks_on()

    l_cax.text(0.5, 0.5, r'$\left[{\rm mJy \, pixel^{-1}}\right]$',
               ha='center', va='center', transform=l_cax.transAxes,
               color='white', rotation=90.)
    r_cax.text(0.5, 0.5, r'$\left[ {\rm pc \, cm^{-6}} \right]$',
               ha='center', va='center', transform=r_cax.transAxes,
               color='white', rotation=90.)

    for cax in caxes:
        cax.yaxis.set_label_position("right")
        cax.minorticks_on()

    if savefig:
        # TODO: Put this in appropriate place in JetModel class
        # jm.log.add_entry("INFO",
        #                    "Radio plot saved to " + savefig)
        plt.savefig(savefig, bbox_inches='tight', dpi=300)

    if show_plot:
        plt.show()

    return None


def jml_profile_plot(inp: Union['JetModel', 'Pipeline'],
                     ax: matplotlib.axes.Axes = None, show_plot: bool = False,
                     savefig: Union[bool, str] = False):
    """
    Plot ejection profile using matlplotlib5

    Parameters
    ----------
    inp
        Pipeline or JetModel instance from which to plot jml(t)
    ax
        Axis to plot to, default is None in which case new axes/figure instances
        are created
    show_plot
        Whether to show the plot on the display device. Useful for interactive
        console sessions, False by default
    savefig
        Whether to save the figure, False by default. Provide the full path of
        the save file as a str to save.

    Returns
    -------
    None
    """
    from RaJePy.classes import Pipeline, JetModel
    from RaJePy import cnsts
    from RaJePy import _config as cfg

    if hasattr(inp, 'runs'):
        jm = inp.model
        run_years = list(set([_.year * con.year for _ in inp.runs]))
    elif hasattr(inp, 'grid'):
        jm = inp
        run_years = []
    else:
        raise TypeError("arg 'inp' must be either a Pipeline or JetModel "
                        f"instance, not {type(inp)}")

    t_0s = [jm.ejections[_]['t_0'] for _ in jm.ejections]
    hls = [jm.ejections[_]['half_life'] for _ in jm.ejections]

    if len(t_0s) > 0 and len(hls) > 0:
        # Plot either 5 half-lives away from first/last existing burst, or from
        # a year either side of the first/last model time specified (if Pipeline
        # provided as arg, inp)
        t_min = np.nanmin([min(run_years) - 1. if run_years else np.nan,
                           np.nanmin(np.array(t_0s) - 5 * np.array(hls))])
        t_max = np.nanmax([max(run_years) + 1. if run_years else np.nan,
                           np.nanmax(np.array(t_0s) + 5 * np.array(hls))])
    else:
        if not run_years:
            t_min, t_max = 0, 10
        else:
            t_min = np.nanmin([np.nanmin(run_years) - 1., 0.])
            t_max = np.nanmax([np.nanmax(run_years) + 1., 10.])

    times = np.linspace(t_min, t_max, 1000)
    jmls = jm.jml_t(times)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(cfg.plots['dims']['text'],
                                              cfg.plots['dims']['column']))

    ax.plot(times / con.year, jmls * con.year / cnsts.MSOL, ls='-',
            color='blue', lw=1, zorder=3, label=r'$\dot{m}_{\rm jet}$')

    if np.ptp(np.log10(ax.get_ylim())) < 1:
        ax.set_ylim(ax.get_ylim()[0],
                    10 ** np.ceil(np.log10(ax.get_ylim()[0]) + 1.))

    ax.axhline(jm.ss_jml * con.year / 1.98847e30, 0, 1, ls=':',
               color='red', lw=1, zorder=2,
               label=r'$\dot{m}_{\rm jet}^{\rm ss}$')

    ax.axvline(run_years, *ax.get_xlim(), ls='-.',
               color='grey', lw=1, zorder=1,
               label=r'$t_\mathrm{runs}$')

    xunit = u.format.latex.Latex(times).to_string(u.year)
    yunit = u.format.latex.Latex(jmls).to_string(u.solMass * u.year ** -1)

    xunit = r' \left[ ' + xunit.replace('$', '') + r'\right] $'
    yunit = r' \left[ ' + yunit.replace('$', '') + r'\right] $'

    ax.set_xlabel(r"$ t \," + xunit)
    ax.set_ylabel(r"$ \dot{m}_{\rm jet}\," + yunit)
    ax.set_yscale('log')
    ax.set_xlim(np.array([t_min, t_max]) / con.year)

    if savefig:
        plt.savefig(savefig, bbox_inches='tight', dpi=300)

    if show_plot:
        plt.show()

    return fig, ax


def geometry_plot(jm: 'JetModel', show_plot: bool = False,
                  savefig: Union[bool, str] = False):
    """
    Plot ejection profile using matlplotlib

    Parameters
    ----------
    jm
        JetModel instance from which to plot mass/volume slices.
    show_plot
        Whether to show the plot on the display device. Useful for interactive
        console sessions, False by default
    savefig
        Whether to save the figure, False by default. Provide the full path of
        the save file as a str to save.

    Returns
    -------
    matplotlib.figure and matplotlib.AxesSubplot instances instantiated or drawn
    upon
    """
    from RaJePy import _config as cfg

    data = jm.fill_factor

    sum_data_x = np.nansum(data, axis=[_ for _ in (0, 1) if _ != jm.los_axis][0])
    sum_data_y = np.nansum(data, axis=jm.los_axis)
    sum_data_z = np.nansum(data, axis=2)

    plt.close('all')

    cmap = matplotlib.cm.get_cmap("inferno").copy()

    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True,
                           figsize=(cfg.plots["dims"]["text"],
                                    cfg.plots["dims"]["text"] * 0.34))

    plt.subplots_adjust(wspace=0.)

    for a in ax:
        a.set_facecolor(cmap(0.))
        a.plot(0, 0, marker='o', mec=None, mfc='w', ms=2, mew=0)

    ax[0].imshow(np.swapaxes(sum_data_x, 0, 1),
                 cmap=cmap,
                 extent=(np.nanmin(jm.yy), np.nanmax(jm.yy),
                         np.nanmin(jm.zz), np.nanmax(jm.zz)),
                 origin='lower')

    ax[1].imshow(np.swapaxes(sum_data_y, 0, 1),
                 cmap=cmap,
                 extent=(np.nanmin(jm.xx), np.nanmax(jm.xx),
                         np.nanmin(jm.zz), np.nanmax(jm.zz)),
                 origin='lower')

    ax[2].imshow(np.swapaxes(sum_data_z, 0, 1),
                 cmap=cmap,
                 extent=(np.nanmin(jm.xx), np.nanmax(jm.xx),
                         np.nanmin(jm.yy), np.nanmax(jm.yy)),
                 origin='lower')

    grid_lines = 'w:'
    ax[0].plot([np.nanmin(jm.yy), np.nanmax(jm.yy),
                np.nanmax(jm.yy), np.nanmin(jm.yy), np.nanmin(jm.yy)],
               [np.nanmin(jm.zz), np.nanmin(jm.zz),
                np.nanmax(jm.zz), np.nanmax(jm.zz), np.nanmin(jm.zz)],
               grid_lines)
    ax[1].plot([np.nanmin(jm.xx), np.nanmax(jm.xx),
                np.nanmax(jm.xx), np.nanmin(jm.xx), np.nanmin(jm.xx)],
               [np.nanmin(jm.zz), np.nanmin(jm.zz),
                np.nanmax(jm.zz), np.nanmax(jm.zz), np.nanmin(jm.zz)],
               grid_lines)
    ax[2].plot([np.nanmin(jm.xx), np.nanmax(jm.xx),
                np.nanmax(jm.xx), np.nanmin(jm.xx), np.nanmin(jm.xx)],
               [np.nanmin(jm.yy), np.nanmin(jm.yy),
                np.nanmax(jm.yy), np.nanmax(jm.yy), np.nanmin(jm.yy)],
               grid_lines)

    def ann_axes(ax, pos, length=0.1, xlab=r'$x$', ylab=r'$y$', zlab=r'$z$',
                 color='w'):
        offsets = ((1., 0.), (0., 1.), (np.cos(np.pi / 6.), np.sin(np.pi / 6.)))
        for i, (dx, dy) in enumerate(offsets):
            ax.arrow(*pos, dx * length, dy * length, length_includes_head=True,
                     transform=ax.transAxes, color=color, overhang=0.2, lw=1,
                     head_width=0.02)
            ax.annotate((xlab, ylab, zlab)[i],
                        xy=pos, xytext=(pos[0] + dx * length, pos[1] + dy * length),
                        xycoords='axes fraction', textcoords='axes fraction',
                        color=color, ha=('left', 'center', 'left')[i],
                        va=('center', 'bottom', 'bottom')[i])

    units = r"$\left[ \mathrm{au} \right]$"
    ax[0].set_ylabel(units)
    ax[1].set_xlabel(units)

    lim = max(np.abs([np.nanmin([jm.grid]), np.nanmax([jm.grid])]) * 1.2)
    ax[0].set_xlim(-lim, lim)
    ax[0].set_ylim(-lim, lim)

    for i, a in enumerate(ax):
        a.tick_params(which='both', direction='in', color='white', top='True',
                      right='True')
        for spine in ('left', 'right', 'top', 'bottom'):
            a.spines[spine].set_color('white')
        a.minorticks_on()
        a.text(0.95, 0.95, ('a', 'b', 'c')[i], transform=a.transAxes,
               ha='right', va='top', color='w')

    ann_axes(ax[0], (0.05, 0.05), xlab=r'$y$', ylab=r'$z$', zlab=r'$x$')
    ann_axes(ax[1], (0.05, 0.05), xlab=r'$x$', ylab=r'$z$', zlab=r'$y$')
    ann_axes(ax[2], (0.05, 0.05), xlab=r'$x$', ylab=r'$y$', zlab=r'$z$')

    ax[0].set_yticks(ax[0].get_xticks())

    ax[0].set_xlim(-lim, lim)
    ax[0].set_ylim(-lim, lim)

    if savefig:
        plt.savefig(savefig, bbox_inches='tight', dpi=300)
        plt.close('all')

    if show_plot:
        plt.show()

    return fig, ax


def sed_plot(pline: 'Pipeline', plot_time: float,
             plot_reynolds: bool = True,
             savefig: bool = False) -> None:
    import matplotlib.pylab as plt
    from RaJePy.maths import physics as mphys
    from RaJePy import _config as cfg

    freqs, fluxes = [], []
    freqs_imfit, fluxes_imfit, efluxes_imfit = [], [], []
    for idx, run in enumerate(pline.runs):
        if run.year == plot_time:
            if run.completed and run.obs_type == 'continuum':
                # Skymodel fluxes
                flux = run.results['flux']
                fluxes.append(flux)
                freqs.append(run.freq)

                # imfit fluxes
                if run.results['imfit'] is not None:
                    flux_imfit = run.results['imfit']['I']['val']
                    eflux_imfit = run.results['imfit']['Ierr']['val']
                    fluxes_imfit.append(flux_imfit)
                    efluxes_imfit.append(eflux_imfit)
                    freqs_imfit.append(run.freq)

    freqs = np.array(freqs)
    fluxes = np.array(fluxes)

    xlims = (10 ** (np.log10(np.min(freqs)) - 0.5),
             10 ** (np.log10(np.max(freqs)) + 0.5))

    alphas = []
    for n in np.arange(1, len(fluxes)):
        alphas.append(np.log10(fluxes[n] /
                               fluxes[n - 1]) /
                      np.log10(freqs[n] / freqs[n - 1]))

    alphas_imfit, ealphas_imfit = [], []
    for n in np.arange(1, len(fluxes_imfit)):
        alphas_imfit.append(np.log10(fluxes_imfit[n] /
                                     fluxes_imfit[n - 1]) /
                            np.log10(freqs_imfit[n] / freqs_imfit[n - 1]))
        c = np.log(freqs_imfit[n] / freqs_imfit[n - 1])
        ealpha = np.sqrt((efluxes_imfit[n] / (fluxes_imfit[n] * c)) ** 2. +
                         (efluxes_imfit[n - 1] / (
                                 fluxes_imfit[n - 1] * c)) ** 2.)
        ealphas_imfit.append(ealpha)

    l_z = pline.model.nz * pline.model.csize / \
          pline.model.params['target']['dist']

    plt.close('all')

    fig, ax1 = plt.subplots(1, 1, figsize=[cfg.plots["dims"]["column"]] * 2)
    ax2 = ax1.twinx()

    # Alphas are calculated at the middle of two neighbouring frequencies
    # in logarithmic space, hence the need for caclulation of freqs_a,
    # the logarithmic mean of the two frequencies
    freqs_a = [10. ** np.mean(np.log10([f, freqs[i + 1]])) for i, f in
               enumerate(freqs[:-1])]
    freqs_a_imfit = [10. ** np.mean(np.log10([f, freqs_imfit[i + 1]])) for
                     i, f in
                     enumerate(freqs_imfit[:-1])]

    ax2.plot(freqs_a, alphas, color='b', ls='None', mec='b', marker='o',
             mfc='cornflowerblue', lw=2, zorder=2, markersize=5)

    ax2.errorbar(freqs_a, alphas_imfit, yerr=ealphas_imfit, ecolor='b',
                 ls='None', capsize=2)

    freqs_r86 = np.logspace(np.log10(np.min(xlims)),
                            np.log10(np.max(xlims)), 100)
    flux_exp = []
    for freq in freqs_r86:
        f = mphys.flux_expected_r86(pline.model, freq, l_z * 0.5)
        flux_exp.append(f * 2.)  # for biconical jet

    alphas_r86 = []
    for n in np.arange(1, len(freqs_r86)):
        alphas_r86.append(np.log10(flux_exp[n] / flux_exp[n - 1]) /
                          np.log10(freqs_r86[n] / freqs_r86[n - 1]))

    # Alphas are calculated at the middle of two neighbouring frequencies
    # in logarithmic space, hence the need for caclulation of freqs_a_r86
    freqs_a_r86 = [10 ** np.mean(np.log10([f, freqs_r86[i + 1]])) for i, f
                   in
                   enumerate(freqs_r86[:-1])]
    if plot_reynolds:
        ax2.plot(freqs_a_r86, alphas_r86, color='cornflowerblue', ls='--',
                 lw=2, zorder=1)

    ax1.loglog(freqs, fluxes, mec='maroon', ls='None', mfc='r', lw=2,
               zorder=3, marker='o', markersize=5)
    ax1.errorbar(freqs_imfit, fluxes_imfit, yerr=efluxes_imfit, ecolor='r',
                 ls='None', capsize=2)

    if plot_reynolds:
        ax1.loglog(freqs_r86, flux_exp, color='r', ls='-', lw=2,
                   zorder=1)
        ax1.loglog(freqs_r86,
                   mphys.approx_flux_expected_r86(pline.model, freqs_r86) *
                   2., color='gray', ls='-.', lw=2, zorder=1)
    ax1.set_xlim(xlims)
    ax2.set_ylim(-0.2, 2.1)
    equalise_axes(ax1, fix_x=False)

    ax1.set_xlabel(r'$\nu \, \left[ {\rm Hz} \right]$', color='k')
    ax1.set_ylabel(r'$S_\nu \, \left[ {\rm Jy} \right]$', color='k')
    ax2.set_ylabel(r'$\alpha$', color='b')

    ax1.tick_params(which='both', direction='in', top=True)
    ax2.tick_params(which='both', direction='in', color='b')
    ax2.tick_params(axis='y', which='both', colors='b')
    ax2.spines['right'].set_color('b')
    ax2.yaxis.label.set_color('b')
    ax2.minorticks_on()

    title = "Radio SED plot at t={:.0f}yr for jet model '{}'"
    title = title.format(plot_time, pline.model.params['target']['name'])

    png_metadata = cfg.plots['metadata']['png']
    png_metadata["Title"] = title

    pdf_metadata = cfg.plots['metadata']['pdf']
    pdf_metadata["Title"] = title

    if savefig:
        fig.savefig(savefig, bbox_inches='tight', metadata=png_metadata,
                    dpi=300)
        fig.savefig(savefig.replace('png', 'pdf'), bbox_inches='tight',
                    metadata=pdf_metadata, dpi=300)
    return None

