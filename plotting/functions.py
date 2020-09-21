# -*- coding: utf-8 -*-
import numpy as np
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.ticker import AutoLocator, AutoMinorLocator, FuncFormatter
from matplotlib.ticker import MultipleLocator, MaxNLocator

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
    
    x_range = np.ptp(ax.get_xlim())
    y_range = np.ptp(ax.get_ylim())
    if ndims == 3:
        z_range = np.ptp(ax.get_zlim())
    
    if logx:
        x_range = np.ptp(np.log10(ax.get_xlim()))
    if logy:
        y_range = np.ptp(np.log10(ax.get_ylim()))
    if ndims == 3 and logz:
        z_range == np.ptp(np.log10(ax.get_zlim()))

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
        ylims = (10**(np.mean(np.log10(ax.get_ylim())) - r / 2.),
                 10**(np.mean(np.log10(ax.get_ylim())) + r / 2.))
    else:
        ylims = (np.mean(ax.get_ylim()) - r / 2.,
                 np.mean(ax.get_ylim()) + r / 2.)
    ax.set_ylim(ylims)

    if ndims == 3:
        if logz:
            zlims = (10**(np.mean(np.log10(ax.get_zlim())) - r / 2.),
                     10**(np.mean(np.log10(ax.get_zlim())) + r / 2.))
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
        if (e < len(colbar) - 1):
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
            cax.set_yscale('log')#, subsy=minticks if isinstance(minticks, list) else [1, 2, 3, 4, 5, 6, 7, 8, 9])
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
        for dir in ['left', 'bottom', 'top']:
            cax.spines[dir].set_visible(False)

