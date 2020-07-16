#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines the following classes:
- JetModel: Handles all radiative transfer and physical calculations of
physical jet model grid.
- ModelRun: Handles all interactions with CASA and execution of a full run
- Pointing (deprecated)
- PoitingScheme (deprecated)

@author: Simon Purser (simonp2207@gmail.com)
"""
import sys
import os
import time
import pickle

import numpy as np
import astropy.units as u
import scipy.constants as con
import matplotlib.pylab as plt
from astropy.coordinates import SkyCoord
from astropy.io import fits
from scipy.spatial import ConvexHull
from scipy.integrate import dblquad, tplquad
from shutil import get_terminal_size
from matplotlib.colors import LogNorm

from RaJePy import logger
from RaJePy.maths import geometry as mgeom
from RaJePy.maths import physics as mphys
from RaJePy.plotting import functions as pfunc

from warnings import filterwarnings
filterwarnings("ignore", category=RuntimeWarning)


class JetModel:
    """
    Class to handle physical model of an ionised jet from a young stellar object
    """

    @classmethod
    def load_model(cls, model_file):
        """
        Loads model from a saved state (pickled file)

        Parameters
        ----------
        cls : JetModel
            DESCRIPTION.
        model_file : str
            Full path to saved model file.

        Returns
        -------
        new_jm : JetModel
            Instance of JetModel to work with.

        """
        # Get the model parameters from the saved model file
        loaded = pickle.load(open(model_file, 'rb'))

        # Create new JetModel class instance
        new_jm = cls(loaded["params"])

        # If fill factors/projected areas have been previously calculated,
        # assign to new instance
        if loaded['ffs'] is not None:
            new_jm.fill_factor = loaded['ffs']

        if loaded['areas'] is not None:
            new_jm.areas = loaded['areas']

        new_jm.time = loaded['time']

        return new_jm

    def __init__(self, params, verbose=True, log=None):
        """

        Parameters
        ----------
        params : dict
            dictionary containing all necessary parameters for run of TORCH

        verbose : bool
            verbosity in terminal. True for verbosity, False for silence
        """
        if isinstance(params, dict):
            self._params = params
        elif isinstance(params, str):
            if not os.path.exists(params):
                raise FileNotFoundError(params + " does not exist")
            if os.path.dirname(params) not in sys.path:
                sys.path.append(os.path.dirname(params))

            jp = __import__(os.path.basename(params).strip('.py'))
            self._params = jp.params
        else:
            raise TypeError("Supplied arg params must be dict or str")

        # self._dcy = self.params['dcys']['model_dcy']
        self._name = self.params['target']['name']
        self._csize = self.params['grid']['c_size']
        self._log = log

        # Number of voxels in x, y, z
        if self.params['grid']['l_z'] is not None:
            nz = int(np.ceil(self.params['grid']['l_z'] / 2. *
                             self.params['target']['dist'] /
                             self.params['grid']['c_size']))
            nx = int(np.ceil(mgeom.w_yz(0., nz + 1.,
                                        self.params['geometry']['w_0'] /
                                        self.params['grid']['c_size'],
                                        self.params['geometry']['r_0'] /
                                        self.params['grid']['c_size'],
                                        self.params['geometry']['epsilon'])
                             )
                     )
            ny = int(np.ceil(mgeom.w_xz(0., nz + 1.,
                                        self.params['geometry']['w_0'] /
                                        self.params['grid']['c_size'],
                                        self.params['geometry']['r_0'] /
                                        self.params['grid']['c_size'],
                                        self.params['geometry']['epsilon'])
                             )
                     )
            nx *= 2
            ny *= 2
            nz *= 2

        else:
            # Enforce even number of cells in every direction
            nx = (self.params['grid']['n_x'] + 1) // 2 * 2
            ny = (self.params['grid']['n_y'] + 1) // 2 * 2
            nz = (self.params['grid']['n_z'] + 1) // 2 * 2

        self._nx = nx
        self._ny = ny
        self._nz = nz
        self._ff = None
        self._areas = None
        self._grid = None

        mlr = self.params['properties']['n_0'] * 1e6 * np.pi
        mlr *= self.params['properties']['mu'] * mphys.atomic_mass("H")
        mlr *= (self.params['geometry']['w_0'] * con.au) ** 2.
        mlr *= self.params['properties']['v_0'] * 1e3  # kg/s

        self._ss_jml = mlr

        def func(jml):
            def func2(t):
                "Mass loss rate as function of time"
                return jml  # * (t / t)

            return func2

        self._jml_t = func(self._ss_jml)
        self._ejections = {}  # Record of any ejection events
        for idx, ejn_t0 in enumerate(self.params['ejection']['t_0']):
            self.add_ejection_event(ejn_t0 * con.year,
                                    mlr * self.params['ejection']['chi'][idx],
                                    self.params['ejection']['hl'][idx] *
                                    con.year)
        self._time = 0. * con.year

    def __str__(self):
        p = self.params
        h = ['Parameter', 'Value']
        d = [('epsilon', format(p['geometry']['epsilon'], '+.3f')),
             ('q_v', format(p['power_laws']['q_v'], '+.3f')),
             ('q_T', format(p['power_laws']['q_T'], '+.3f')),
             ('q_x', format(p['power_laws']['q_x'], '+.3f')),
             ('q_n', format(p['power_laws']['q_n'], '+.3f')),
             ('q_tau', format(p['power_laws']['q_tau'], '+.3f')),
             ('w_0', format(p['geometry']['w_0'], '.2f') + ' au'),
             ('r_0', format(p['geometry']['r_0'], '.2f') + ' au'),
             ('v_0', format(p['properties']['v_0'], '.0f') + ' km/s'),
             ('x_0', format(p['properties']['x_0'], '.3f')),
             ('n_0', format(p['properties']['n_0'], '.3e') + ' cm^-3'),
             ('T_0', format(p['properties']['T_0'], '.0e') + ' K'),
             ('i', format(p['geometry']['inc'], '+.1f') + ' deg'),
             ('theta', format(p['geometry']['pa'], '+.1f') + ' deg'),
             ('cross-section',
              'exp' if p['geometry']['exp_cs'] else 'constant'),
             ('t_now', format(self.time / con.year, '+.3f') + ' yr')]

        for idx, t_0 in enumerate(p['ejection']['t_0']):
            d.append(("t0_" + str(idx + 1), format(t_0, '+.3f') + ' yr'))
            d.append(("hl_" + str(idx + 1),
                      format(p['ejection']['hl'][idx], '.3f') + ' yr'))
            d.append(("chi_" + str(idx + 1),
                      format(p['ejection']['chi'][idx], '.3f')))

        col1_width = max(map(len, [h[0]] + list(list(zip(*d))[0]))) + 2
        col2_width = max(map(len, [h[1]] + list(list(zip(*d))[1]))) + 2
        tab_width = col1_width + col2_width + 3

        hline = tab_width * '-'
        delim = '|'

        s = format('JET MODEL', '^' + str(tab_width)) + '\n'
        s += hline + '\n'
        s += delim + delim.join([format(h[0], '^' + str(col1_width)),
                                 format(h[1], '^' + str(col2_width))]) + delim
        s += '\n' + hline + '\n'
        for l in d:
            s += delim + delim.join([format(l[0], '^' + str(col1_width)),
                                     format(l[1], '^' + str(col2_width))]) + \
                 delim + '\n'
        s += hline + '\n'
        return s

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, new_time):
        self._time = new_time

    @property
    def jml_t(self):
        return self._jml_t

    @jml_t.setter
    def jml_t(self, new_jml_t):
        self._jml_t = new_jml_t

    def add_ejection_event(self, t_0, peak_jml, half_life):
        """
        Add ejection event in the form of a Gaussian ejection profile as a
        function of time

        Parameters
        ----------
        t_0 : astropy.units.quantity.Quantity
            Time of peak mass loss rate
        peak_jml : astropy.units.quantity.Quantity
            Highest jet mass loss rate of ejection burst
        half_life : astropy.units.quantity.Quantity
            Time for mass loss rate to halve during the burst

        Returns
        -------
        None.

        """

        def func(fnc, t_0, peak_jml, half_life):
            """

            Parameters
            ----------
            fnc : Time dependent function giving current jet mass loss rate
            t_0 : Time of peak of burst
            peak_jml : Peak of burst's jet mass loss rate
            half_life : FWHM of burst

            Returns
            -------
            Factory function returning function describing new time dependent
            mass loss rate incorporating input burst

            """
            def func2(t):
                """Gaussian profiled ejection event"""
                amp = peak_jml - self._ss_jml
                sigma = half_life * 2. / (2. * np.sqrt(2. * np.log(2.)))
                return fnc(t) + amp * np.exp(-(t - t_0) ** 2. /
                                             (2. * sigma ** 2.))

            return func2

        self._jml_t = func(self._jml_t, t_0, peak_jml, half_life)

        record = {'t_0': t_0, 'peak_jml': peak_jml, 'half_life': half_life}
        self._ejections[str(len(self._ejections) + 1)] = record

    @property
    def grid(self):
        if self._grid:
            return self._grid
        self._grid = np.meshgrid(np.linspace(-self.nx / 2 * self.csize,
                                             (self.nx / 2 - 1.) * self.csize,
                                             self.nx),
                                 np.linspace(-self.ny / 2 * self.csize,
                                             (self.ny / 2 - 1.) * self.csize,
                                             self.ny),
                                 np.linspace(-self.nz / 2 * self.csize,
                                             (self.nz / 2 - 1.) * self.csize,
                                             self.nz))

        return self._grid

    @grid.setter
    def grid(self, new_grid):
        self._grid = new_grid

    @property
    def fill_factor(self):
        """
        Calculate the fraction of each of the grid's cells falling within the
        jet
        """
        if self._ff is not None:
            return self._ff

        # Set up coordinate grid in x, y, z for grid's 1st octant only due to
        # assumption of reflective symmetry about x, y and z axes
        xx, yy, zz = [_[int(self.ny / 2):,
                      int(self.nx / 2):,
                      int(self.nz / 2):] for _ in self.grid]

        nvoxels = (self.nx / 2) * (self.ny / 2) * (self.nz / 2)

        if self.log:
            self._log.add_entry(mtype="INFO",
                                entry="Calculating cells' fill "
                                      "factors/projected areas")

        else:
            print("INFO: Calculating cells' fill factors/projected areas")

        # Assign to local variables for readability
        w_0 = self.params['geometry']['w_0']
        r_0 = self.params['geometry']['r_0']
        eps = self.params['geometry']['epsilon']
        cs = self.csize

        w_0_cs = w_0 / cs
        r_0_cs = r_0 / cs

        def hfun_area(n_z):
            def func(x):
                """
                The upper boundary surface in z for area calculation
                """
                return n_z + 1.

            return func

        def gfun_area(w_0, n_y, n_z, r_0, eps):
            """
            The lower boundary curve in z for area calculation
            """

            def func(x):
                return np.max([r_0, n_z,
                               r_0 * w_0 ** (-1. / eps) *
                               (x ** 2. + n_y ** 2.) ** (1. / (2. * eps))])

            return func

        def hfun(w_0, n_y, n_z, r_0, eps):
            """
            The upper boundary curve in y for volume calculation
            """

            def func(x):
                return np.min([np.sqrt(w_0 ** 2. *
                                       ((n_z + 1.) / r_0) ** (
                                               2 * eps) - x ** 2.),
                               n_y + 1])

            return func

        def gfun(n_y):
            def func(_):
                """
                The lower boundary curve in y for volume calculation
                """
                return n_y

            return func

        def rfun(n_z):
            def func(_, __):
                """
                The upper boundary surface in z for volume calculation
                """
                return n_z + 1.

            return func

        def qfun(w_0, n_z, r_0, eps):
            """
            The lower boundary surface in z for volume calculation
            """

            def func(x, y):
                return np.max([r_0 * w_0 ** (-1. / eps) *
                               (x ** 2. + y ** 2.) ** (1. / (2. * eps)),
                               r_0, n_z])

            return func

        ffs = np.zeros(np.shape(xx))
        areas = np.zeros(np.shape(xx))  # Areas as projected on to the y-axis
        count = 0
        progress = -1
        then = time.time()

        for idxy, yplane in enumerate(zz):
            for idxx, xrow in enumerate(yplane):
                for idxz, z in enumerate(xrow):
                    count += 1
                    x, y = xx[idxy][idxx][idxz], yy[idxy][idxx][idxz]
                    verts = ((x, y, z), (x + cs, y, z),
                             (x, y + cs, z), (x + cs, y + cs, z),
                             (x, y, z + cs), (x + cs, y, z + cs),
                             (x, y + cs, z + cs),
                             (x + cs, y + cs, z + cs))
                    verts_inside = []
                    for vert in verts:
                        zjet = mgeom.w_xy(vert[0], vert[1], w_0, r_0, eps)
                        if vert[2] > zjet:
                            verts_inside.append(True)
                        else:
                            verts_inside.append(False)

                    if sum(verts_inside) == 0:
                        continue
                    elif sum(verts_inside) == 8:
                        ff = 1.
                        area = 1.
                    else:
                        # Calculate tuple of possible intercept coordinates for
                        # every edge of the voxel
                        i = ((mgeom.w_yz(verts[0][1], verts[0][2],
                                         w_0, r_0, eps),
                              verts[0][1],
                              verts[0][2]),  # a
                             (verts[0][0],
                              mgeom.w_xz(verts[0][0], verts[0][2],
                                         w_0, r_0, eps),
                              verts[0][2]),  # b
                             (verts[0][0],
                              verts[0][1],
                              mgeom.w_xy(verts[0][0], verts[0][1],
                                         w_0, r_0, eps)),  # c
                             (verts[1][0],
                              mgeom.w_xz(verts[1][0], verts[1][2],
                                         w_0, r_0, eps),
                              verts[1][2]),  # d
                             (verts[1][0],
                              verts[1][1],
                              mgeom.w_xy(verts[1][0], verts[1][1],
                                         w_0, r_0, eps)),  # e
                             (mgeom.w_yz(verts[2][1], verts[2][2],
                                         w_0, r_0, eps),
                              verts[2][1],
                              verts[2][2]),  # f
                             (verts[2][0],
                              verts[2][1],
                              mgeom.w_xy(verts[2][0], verts[2][1],
                                         w_0, r_0, eps)),  # g
                             (verts[3][0],
                              verts[3][1],
                              mgeom.w_xy(verts[3][0], verts[3][1],
                                         w_0, r_0, eps)),  # h
                             (mgeom.w_yz(verts[4][1], verts[4][2],
                                         w_0, r_0, eps),
                              verts[4][1],
                              verts[4][2]),  # i
                             (verts[4][0],
                              mgeom.w_xz(verts[4][0], verts[4][2],
                                         w_0, r_0, eps),
                              verts[4][2]),  # j
                             (verts[5][0],
                              mgeom.w_xz(verts[5][0], verts[5][2],
                                         w_0, r_0, eps),
                              verts[5][2]),  # k
                             (mgeom.w_yz(verts[6][1], verts[6][2],
                                         w_0, r_0, eps),
                              verts[6][1],
                              verts[6][2]))  # l

                        # Determine which edges actually have an intercept with jet
                        # boundary
                        # NB - ^ is the bitwise operator XOR
                        mask = [verts_inside[0] ^ verts_inside[1],
                                verts_inside[0] ^ verts_inside[2],
                                verts_inside[0] ^ verts_inside[4],
                                verts_inside[1] ^ verts_inside[3],
                                verts_inside[1] ^ verts_inside[5],
                                verts_inside[2] ^ verts_inside[3],
                                verts_inside[2] ^ verts_inside[6],
                                verts_inside[3] ^ verts_inside[7],
                                verts_inside[4] ^ verts_inside[5],
                                verts_inside[4] ^ verts_inside[6],
                                verts_inside[5] ^ verts_inside[7],
                                verts_inside[6] ^ verts_inside[7]]

                        # Create array of coordinates defining the polygon's
                        # vertices lying within the jet boundary
                        ph_verts = np.append(
                            np.array(verts)[verts_inside].tolist(),
                            np.array(i)[mask].tolist(),
                            axis=0)
                        # Use scipy's ConvexHull's built in method to determine
                        # volume within jet boundary (MUST BE CONVEX POLYGON)
                        try:
                            ch_vol = ConvexHull(ph_verts)
                            ch_area = ConvexHull(ph_verts.T[::2].T)
                            ff = ch_vol.volume / cs ** 3.
                            area = ch_area.volume / cs ** 2.
                        except:
                            x /= cs
                            y /= cs
                            z /= cs
                            b = np.min([np.sqrt(w_0_cs ** 2. *
                                                ((z + 1.) / r_0_cs) **
                                                (2. * eps) - y ** 2.),
                                        x + 1.])
                            ff = tplquad(lambda z, y, x: 1.,
                                         a=x, b=b,
                                         gfun=gfun(y),
                                         hfun=hfun(w_0_cs, y,
                                                   z, r_0_cs, eps),
                                         qfun=qfun(w_0_cs, z,
                                                   r_0_cs, eps),
                                         rfun=rfun(z))[0]

                            b_a = np.sqrt(w_0_cs ** 2. *
                                          ((z + 1.) /
                                           r_0_cs) ** (2. * eps) - y ** 2.)
                            area = dblquad(lambda z, x: 1.,
                                           a=x, b=np.min([b_a, x + 1.]),
                                           gfun=gfun_area(w_0_cs, y, z, r_0_cs,
                                                          eps),
                                           hfun=hfun_area(z))[0]
                            x *= cs
                            y *= cs
                            z *= cs

                        # Accurately calculate filling fractions for 5 base
                        # cell layers
                        if ((np.round(z / cs) - np.round(r_0_cs) < 5) and
                                (np.round(z / cs) - np.round(r_0_cs) > -1)):
                            x /= cs
                            y /= cs
                            z /= cs

                            # Volume
                            b = np.min([np.sqrt(w_0_cs ** 2. *
                                                ((z + 1.) / r_0_cs) **
                                                (2. * eps) - y ** 2.),
                                        x + 1.])
                            ff = tplquad(lambda z, y, x: 1.,
                                         a=x, b=b,
                                         gfun=gfun(y),
                                         hfun=hfun(w_0_cs, y, z, r_0_cs, eps),
                                         qfun=qfun(w_0_cs, z, r_0_cs, eps),
                                         rfun=rfun(z))[0]

                            b_a = np.sqrt(w_0_cs ** 2. *
                                          ((z + 1.) / r_0_cs) **
                                          (2. * eps) - y ** 2.)

                            area = dblquad(lambda z, x: 1.,
                                           a=x, b=np.min([b_a, x + 1.]),
                                           gfun=gfun_area(w_0_cs, y, z, r_0_cs,
                                                          eps),
                                           hfun=hfun_area(z))[0]
                    ffs[idxy][idxx][idxz] = ff
                    areas[idxy][idxx][idxz] = area

                # Progress bar
                new_progress = int(count / nvoxels * 100)  # 
                if new_progress > progress:
                    progress = new_progress
                    pblen = get_terminal_size().columns - 1
                    pblen -= 16  # 16 non-varying characters
                    s = '[' + ('=' * (int(progress / 100 * pblen) - 1)) + \
                        ('>' if int(progress / 100 * pblen) > 0 else '') + \
                        (' ' * int(pblen - int(progress / 100 * pblen))) + '] '
                    # s += format(int(progress), '3') + '% complete'
                    if progress != 0.:
                        t_sofar = (time.time() - then)
                        rate = progress / t_sofar
                        s += time.strftime('%Hh%Mm%Ss left',
                                           time.gmtime(
                                               (100. - progress) / rate))
                    else:
                        s += '  h  m  s left'
                    print('\r' + s, end='' if progress < 100 else '\n')

        now = time.time()
        if self.log:
            self.log.add_entry(mtype="INFO",
                               entry=time.strftime('Finished in %Hh%Mm%Ss',
                                                   time.gmtime(now - then)))
        else:
            print(time.strftime('INFO: Finished in %Hh%Mm%Ss',
                                time.gmtime(now - then)))

        # Reflect in x, y and z axes
        for ax in (0, 1, 2):
            ffs = np.append(np.flip(ffs, axis=ax), ffs, axis=ax)
            areas = np.append(np.flip(areas, axis=ax), areas, axis=ax)

        # Included as there are some, presumed floating point errors giving
        # fill factors of ~1e-15 on occasion
        ffs = np.where(ffs > 1e-6, ffs, np.NaN)
        areas = np.where(areas > 1e-6, areas, np.NaN)

        self._ff = ffs
        self._areas = areas

        return self._ff

    @fill_factor.setter
    def fill_factor(self, new_ffs):
        self._ff = new_ffs

    @property
    def areas(self):
        """
        Areas of jet-filled portion of cells as projected on to the y-axis
        (hopefully, custom orientations will address this so area is as 
        projected on to a surface whose normal points to the observer)
        """
        if "_areas" in self.__dict__.keys() and self._areas is not None:
            return self._areas
        else:
            self.fill_factor  # Areas calculated as part of fill factors

        return self._areas

    @areas.setter
    def areas(self, new_areas):
        self._areas = new_areas

    def save(self, filename):
        ps = {'params': self._params,
              'areas': None if self._areas is None else self.areas,
              'ffs': None if self._ff is None else self.fill_factor,
              'time': self.time}
        pickle.dump(ps, open(filename, "wb"))
        return None

    @property
    def mass(self):
        if hasattr(self, '_m'):
            return self._m * self.chi_xyz

        w_0 = self.params['geometry']['w_0'] / self.params['grid']['c_size']
        r_0 = self.params['geometry']['r_0'] / self.params['grid']['c_size']
        eps = self.params['geometry']['epsilon']

        # mlr = self.params['properties']['n_0'] * 1e6 * np.pi
        # mlr *= self.params['properties']['mu'] * mphys.atomic_mass("H")
        # mlr *= (self.params['geometry']['w_0'] * con.au)**2.
        # mlr *= self.params['properties']['v_0'] * 1e3  # kg/s

        # Mass of slice with z-width == 1 full cell
        mass_full_slice = self._ss_jml * (self.csize * con.au /  # kg
                                          (self.params['properties'][
                                               'v_0'] * 1e3))

        ms = np.zeros(np.shape(self.fill_factor))
        constant = np.pi * w_0 ** 2. / ((2. * eps + 1.) * r_0 ** (2. * eps))

        for idz, z in enumerate(self.grid[2][0][0] / self.csize):
            z = np.round(z)
            n_z = int(np.min(np.abs([z, z + 1])))
            if n_z > r_0:
                vol_zlayer = constant * ((n_z + 1.) ** (2. * eps + 1) -
                                         (n_z + 0.) ** (2. * eps + 1))
                mass_slice = mass_full_slice
            elif (n_z + 1) >= r_0:
                vol_zlayer = constant * ((n_z + 1.) ** (2. * eps + 1) -
                                         r_0 ** (2. * eps + 1))
                mass_slice = mass_full_slice * (n_z + 1. - r_0)
            else:
                vol_zlayer = 0.
                mass_slice = 0.
                continue

            ffs_zlayer = self.fill_factor[:, :, idz]
            m_cell = mass_slice / vol_zlayer  # kg / cell
            ms_zlayer = ffs_zlayer * m_cell

            ms[:, :, idz] = ms_zlayer

        ms = np.where(self.fill_factor > 0, ms, np.NaN)

        self.mass = ms
        return self._m * self.chi_xyz

    @mass.setter
    def mass(self, new_ms):
        self._m = new_ms * self.chi_xyz

    @property
    def chi_xyz(self):
        """
        Chi factor (the burst factor) as a function of position.
        """
        z = np.abs(self.grid[2] + 0.5 * self.csize)
        a = z - 0.5 * self.csize
        b = z + 0.5 * self.csize

        a = np.where(b <= self.params['geometry']['r_0'], np.NaN, a)
        b = np.where(b <= self.params['geometry']['r_0'], np.NaN, b)

        a = np.where(a <= self.params['geometry']['r_0'],
                     self.params['geometry']['r_0'], a)

        z *= con.au
        a *= con.au
        b *= con.au

        def t_z(z):
            """
            Time as a function of z. Defined purely for informative purposes
            """
            r_0 = self.params['geometry']['r_0'] * con.au
            v_0 = self.params['properties']['v_0'] * 1000
            q_v = self.params['power_laws']['q_v']
            return (r_0 ** q_v * z ** (1. - q_v) - r_0) / (v_0 * (1. - q_v))

        def int_t_z(z):
            """
            Integral of t_z defined above for use in average value finding
            """
            r_0 = self.params['geometry']['r_0'] * con.au
            v_0 = self.params['properties']['v_0'] * 1000.
            q_v = self.params['power_laws']['q_v']
            num = r_0 ** q_v * z ** (2. - q_v) + (q_v - 2.) * r_0 * z
            den = v_0 * (q_v - 2.) * (q_v - 1.)
            return num / den

        av_ts = 1. / (b - a)
        av_ts *= int_t_z(b) - int_t_z(a)

        # So that times start at 0 at r_0 and to progress to current model time
        av_ts = self.time - av_ts

        av_ts = np.where(self.fill_factor > 0, av_ts, np.NaN)

        av_chis = self._jml_t(av_ts) / self._ss_jml

        return av_chis

    @property
    def number_density(self):
        if hasattr(self, '_nd'):
            return self._nd * self.chi_xyz

        z = np.abs(self.grid[2] + 0.5 * self.csize)
        a = z - 0.5 * self.csize
        b = z + 0.5 * self.csize

        a = np.where(b <= self.params['geometry']['r_0'], np.NaN, a)
        b = np.where(b <= self.params['geometry']['r_0'], np.NaN, b)

        a = np.where(a <= self.params['geometry']['r_0'],
                     self.params['geometry']['r_0'], a)

        # Method 1, i.e. via Reynolds (1986) power-law for n(r) and
        # averaging cell number density over z-axis extent of each cell. See
        # https://www.math24.net/average-value-function/ for math
        nd = self.params['properties']['n_0']
        nd *= self.params['geometry']['r_0'] ** -self.params["power_laws"][
            "q_n"]
        nd *= (b ** (self.params["power_laws"]["q_n"] + 1) -
               a ** (self.params["power_laws"]["q_n"] + 1))
        nd /= self.params["power_laws"]["q_n"] + 1
        nd /= self.csize

        nd = np.where(self.fill_factor > 0, nd, np.NaN)

        # nd = self.mass / (self.fill_factor * (jm.csize * con.au)**3.)
        # nd /=  self.params['properties']['mu'] * mphys.atomic_mass("H")
        # nd /= 1e6  # m^-3 to cm^-3

        self.number_density = np.nan_to_num(nd, nan=np.NaN, posinf=np.NaN,
                                            neginf=np.NaN)

        return self._nd * self.chi_xyz

    @number_density.setter
    def number_density(self, new_nds):
        self._nd = new_nds

    @property
    def mass_density(self):
        """
        Mass density in g cm^-3
        """
        mean_m_particle = self.params['properties']['mu'] * \
                          mphys.atomic_mass("H")
        return mean_m_particle * 1e3 * self.number_density

    @property
    def ion_fraction(self):
        if hasattr(self, '_xi'):
            return self._xi
        z = np.abs(self.grid[2] + 0.5 * self.csize)
        a = z - 0.5 * self.csize
        b = z + 0.5 * self.csize

        a = np.where(b <= self.params['geometry']['r_0'], np.NaN, a)
        b = np.where(b <= self.params['geometry']['r_0'], np.NaN, b)

        a = np.where(a <= self.params['geometry']['r_0'],
                     self.params['geometry']['r_0'], a)

        # Averaging cell ionisation fraction over z-axis extent of each cell.
        # See https://www.math24.net/average-value-function/ for math
        xi = self.params['properties']['x_0']
        xi *= self.params['geometry']['r_0'] ** -self.params["power_laws"][
            "q_x"]
        xi *= (b ** (self.params["power_laws"]["q_x"] + 1) -
               a ** (self.params["power_laws"]["q_x"] + 1))
        xi /= self.params["power_laws"]["q_x"] + 1
        xi /= self.csize

        xi = np.where(self.fill_factor > 0., xi, np.NaN)

        self.ion_fraction = xi

        return self._xi

    @ion_fraction.setter
    def ion_fraction(self, new_xis):
        self._xi = new_xis

    # @property
    def emission_measure(self, savefits=False):
        ems = (self.number_density * self.ion_fraction) ** 2. * \
              (self.csize * con.au / con.parsec *
               (self.fill_factor / self.areas))

        from scipy.ndimage import rotate
        ems = rotate(ems, axes=(2, 0), reshape=True, order=0, prefilter=False,
                     angle=90. - self.params['geometry']['inc'])
        ems = rotate(ems, axes=(2, 1), reshape=True, order=0, prefilter=False,
                     angle=self.params['geometry']['pa'])

        ems = np.nansum(ems, axis=0)
        # self.emission_measure = ems

        if savefits:
            self.save_fits(ems.T, savefits, 'em')

        return ems

    # @emission_measure.setter
    # def emission_measure(self, new_ems):
    #     self._em = new_ems

    def optical_depth_ff(self, freq, savefits=False):
        """
        Return free-free optical depth as viewed along the y-axis

        Parameters
        ----------
        freq : float
            Frequency of observation (Hz).
        savefits : bool, str
            False or full path to save calculated optical depths as .fits file

        Returns
        -------
        tau_ff : numpy.ndarray
            Optical depths as viewed along y-axis.

        """
        # Gaunt factors of van Hoof et al. (2014). Use if constant temperature
        # as computation via this method across a grid takes too long
        if self.params['power_laws']['q_T'] == 0.1241:
            gff = mphys.gff(freq, self.params['properties']['T_0'])

        # Equation 1 of Reynolds (1986) otherwise as an approximation
        else:
            gff = 11.95 * self.temperature ** 0.15 * freq ** -0.1

        # Equation 1.26 and 5.19b of Rybicki and Lightman (cgs). Averaged
        # path length through voxel is volume / projected area
        tff = 0.018 * self.temperature ** -1.5 * freq ** -2. * \
              (self.number_density * self.ion_fraction) ** 2. * \
              (self.csize * con.au * 1e2 * \
               (self.fill_factor / self.areas)) * gff

        from scipy.ndimage import rotate
        tff = rotate(tff, axes=(2, 0), reshape=True, order=0, prefilter=False,
                     angle=90. - self.params['geometry']['inc'])
        tff = rotate(tff, axes=(2, 1), reshape=True, order=0, prefilter=False,
                     angle=self.params['geometry']['pa'])

        tau_ff = np.nansum(tff, axis=0)

        if savefits:
            self.save_fits(tau_ff.T, savefits, 'tau', freq)

        return tau_ff

    def intensity_ff(self, freq):
        """
        Radio intensity as viewed along x-axis (in W m^-2 Hz^-1 sr^-1)
        """

        from scipy.ndimage import rotate
        ts = rotate(self.temperature, axes=(2, 0), reshape=True, order=0,
                    prefilter=False,
                    angle=90. - self.params['geometry']['inc'])
        ts = rotate(ts, axes=(2, 1), reshape=True, order=0, prefilter=False,
                    angle=self.params['geometry']['pa'])

        T_b = np.nanmean(np.where(ts > 0., ts, np.NaN), axis=0) * \
              (1. - np.exp(-self.optical_depth_ff(freq)))

        ints = 2. * freq ** 2. * con.k * T_b / con.c ** 2.

        return ints

    def flux_ff(self, freq, savefits=False):
        """
        Return flux (in Jy)
        """
        ints = self.intensity_ff(freq)
        fluxes = ints * np.tan((self.csize * con.au) /
                               (self.params["target"]["dist"] *
                                con.parsec)) ** 2. / 1e-26

        if savefits:
            self.save_fits(fluxes.T, savefits, 'flux', freq)

        return fluxes

    def save_fits(self, data, filename, image_type, freq=None):
        """
        Save .fits file of input data

        Parameters
        ----------
        data : numpy.array
            2-D numpy array of image data.
        filename: str
            Full path to save .fits image to
        image_type : str
            One of 'flux', 'tau' or 'em'. The type of image data saved.
        freq : float
            Radio frequency of image (ignored if image_type is 'em')

        Returns
        -------
        None.

        """
        if image_type not in ('flux', 'tau', 'em'):
            raise ValueError("arg image_type must be one of 'flux', 'tau' or "
                             "'em'")

        c = SkyCoord(self.params['target']['ra'],
                     self.params['target']['dec'],
                     unit=(u.hourangle, u.degree), frame='fk5')

        csize_deg = np.degrees(np.arctan(self.csize * con.au /
                                         (self.params['target']['dist'] *
                                          con.parsec)))

        hdu = fits.PrimaryHDU(np.array([data]))
        hdul = fits.HDUList([hdu])
        hdr = hdul[0].header

        hdr['AUTHOR'] = 'S.J.D.Purser'
        hdr['OBJECT'] = self.params['target']['name']
        hdr['CTYPE1'] = 'RA---TAN'
        hdr.comments['CTYPE1'] = 'x-coord type is RA Tan Gnomonic projection'
        hdr['CTYPE2'] = 'DEC--TAN'
        hdr.comments['CTYPE2'] = 'y-coord type is DEC Tan Gnomonic projection'
        hdr['EQUINOX'] = 2000.
        hdr.comments['EQUINOX'] = 'Equinox of coordinates'
        hdr['CRPIX1'] = self.nx / 2 + 0.5
        hdr.comments['CRPIX1'] = 'Reference pixel in RA'
        hdr['CRPIX2'] = self.nz / 2 + 0.5
        hdr.comments['CRPIX2'] = 'Reference pixel in DEC'
        hdr['CRVAL1'] = c.ra.deg
        hdr.comments['CRVAL1'] = 'Reference pixel value in RA (deg)'
        hdr['CRVAL2'] = c.dec.deg
        hdr.comments['CRVAL2'] = 'Reference pixel value in DEC (deg)'
        hdr['CDELT1'] = -csize_deg
        hdr.comments['CDELT1'] = 'Pixel increment in RA (deg)'
        hdr['CDELT2'] = csize_deg
        hdr.comments['CDELT2'] = 'Pixel size in DEC (deg)'

        if image_type in ('flux', 'tau'):
            hdr['CDELT3'] = 1.
            hdr.comments['CDELT3'] = 'Frequency increment (Hz)'
            hdr['CRPIX3'] = 0.5
            hdr.comments['CRPIX3'] = 'Reference frequency (channel number)'
            hdr['CRVAL3'] = freq
            hdr.comments['CRVAL3'] = 'Reference frequency (Hz)'

        if image_type == 'flux':
            hdr['BUNIT'] = 'Jy/pixel'
        elif image_type == 'em':
            hdr['BUNIT'] = 'pc cm^-6'
        else:
            hdr['BUNIT'] = 'dimensionless'

        s_hist = self.__str__().split('\n')
        hdr['HISTORY'] = (' ' * (72 - len(s_hist[0]))).join(s_hist)

        hdul.writeto(filename, overwrite=True)

        return None

    @property
    def temperature(self):
        """
        Temperature (in Kelvin)
        """
        if hasattr(self, '_t'):
            return self._t
        z = np.abs(self.grid[2] + 0.5 * self.csize)
        a = z - 0.5 * self.csize
        b = z + 0.5 * self.csize

        a = np.where(b <= self.params['geometry']['r_0'], np.NaN, a)
        b = np.where(b <= self.params['geometry']['r_0'], np.NaN, b)

        a = np.where(a <= self.params['geometry']['r_0'],
                     self.params['geometry']['r_0'], a)

        # Averaging cell temperature over z-axis extent of each cell. See
        # https://www.math24.net/average-value-function/ for math
        ts = self.params['properties']['T_0']
        ts *= self.params['geometry']['r_0'] ** -self.params["power_laws"][
            "q_T"]
        ts *= (b ** (self.params["power_laws"]["q_T"] + 1) -
               a ** (self.params["power_laws"]["q_T"] + 1))
        ts /= self.params["power_laws"]["q_T"] + 1
        ts /= self.csize

        ts = np.where(self.fill_factor > 0., ts, np.NaN)
        self.temperature = ts

        return self._t

    @temperature.setter
    def temperature(self, new_ts):
        self._t = new_ts

    @property
    def pressure(self):
        """
        Pressure in Barye (or dyn cm^-2)
        """
        return self.number_density * self.temperature * con.k * 1e7

    @property
    def vel(self):
        """
        Velocity components in km/s
        """
        if hasattr(self, '_v'):
            return self._v

        x = self.grid[0] + 0.5 * self.csize
        y = self.grid[1] + 0.5 * self.csize
        z = np.abs(self.grid[2] + 0.5 * self.csize)

        r = np.sqrt(x ** 2. + y ** 2.)

        w_0 = self.params['geometry']['w_0']
        r_0 = self.params['geometry']['r_0']
        eps = self.params['geometry']['epsilon']
        m1 = self.params['target']['m_star'] * 1.98847e30  # kg    

        a = z - 0.5 * self.csize
        b = z + 0.5 * self.csize

        a = np.where(b <= r_0, np.NaN, a)
        b = np.where(b <= r_0, np.NaN, b)

        a = np.where(a <= r_0, r_0, a)

        # Averaging cell z-velocity over z-axis extent of each cell. See
        # https://www.math24.net/average-value-function/ for math
        vz = self.params['properties']['v_0']
        vz *= self.params['geometry']['r_0'] ** -self.params["power_laws"][
            "q_v"]
        vz *= (b ** (self.params["power_laws"]["q_v"] + 1) -
               a ** (self.params["power_laws"]["q_v"] + 1))
        vz /= self.params["power_laws"]["q_v"] + 1
        vz /= self.csize

        # Effective radius of (x, y) point in jet stream i.e. from what radius
        # in the disc the material was launched
        r_eff = (r_0 ** eps * (self.params['target']['r_2'] -
                               self.params['target']['r_1']) * r / \
                 (w_0 * z ** eps)) + self.params['target']['r_1']

        vx = np.sqrt(con.G * m1 / (r_eff * con.au)) * np.sin(np.arctan2(y, x))
        vy = np.sqrt(con.G * m1 / (r_eff * con.au)) * np.cos(np.arctan2(y, x))

        vx /= 1e3  # km/s
        vy /= 1e3  # km/s

        # vx = -vx here because velocities appear flipped in checks
        vx = -np.where(self.fill_factor > 0., vx, np.NaN)
        vy = np.where(self.fill_factor > 0., vy, np.NaN)
        vz = np.where(self.grid[2] > 0, vz, -vz)
        vz = np.where(self.fill_factor > 0., vz, np.NaN)

        self.vel = (vx, vy, vz)

        return self._v

    @vel.setter
    def vel(self, new_vs):
        self._v = new_vs

    @property
    def log(self):
        return self._log

    @log.setter
    def log(self, new_log):
        self._log = new_log

    @property
    def csize(self):
        return self._csize

    @property
    def nx(self):
        return self._nx

    @property
    def ny(self):
        return self._ny

    @property
    def nz(self):
        return self._nz

    @property
    def params(self):
        return self._params

    @property
    def name(self):
        return self._name

    def model_plot(self, savefig=False):
        """
        Generate 4 subplots of (from top left, clockwise) number density,
        temperature, ionisation fraction and velocity.


        Parameters
        ----------
        savefig: bool, str
            Whether to save the radio plot to file. If False, will not, but if
            a str representing a valid path will save to that path.

        Returns
        -------
        None.

        """
        import matplotlib.gridspec as gridspec

        plt.close('all')

        fig = plt.figure(figsize=(6.65, 6.65))

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

        # Velocity z-component
        br_ax = plt.subplot(br_cell[0, 0])
        br_cax = plt.subplot(br_cell[0, 1])

        bbox = tl_ax.get_window_extent()
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
        aspect = bbox.width / bbox.height

        im_nd = tl_ax.imshow(self.number_density[self.ny // 2].T,
                             norm=LogNorm(vmin=np.nanmin(self.number_density),
                                          vmax=np.nanmax(self.number_density)),
                             extent=(np.min(self.grid[1]),
                                     np.max(self.grid[1]) + self.csize * 1.,
                                     np.min(self.grid[2]),
                                     np.max(self.grid[2]) + self.csize * 1.),
                             cmap='viridis_r', aspect="equal")
        tl_ax.set_xlim(np.array(tl_ax.get_ylim()) * aspect)
        pfunc.make_colorbar(tl_cax, np.nanmax(self.number_density),
                            cmin=np.nanmin(self.number_density),
                            position='right', orientation='vertical',
                            numlevels=50, colmap='viridis_r', norm=im_nd.norm)

        im_T = tr_ax.imshow(self.temperature[self.ny // 2].T,
                            norm=LogNorm(vmin=100.,
                                         vmax=max([1e4, np.nanmax(
                                             self.temperature)])),
                            extent=(np.min(self.grid[1]),
                                    np.max(self.grid[1]) + self.csize * 1.,
                                    np.min(self.grid[2]),
                                    np.max(self.grid[2]) + self.csize * 1.),
                            cmap='plasma', aspect="equal")
        tr_ax.set_xlim(np.array(tr_ax.get_ylim()) * aspect)
        pfunc.make_colorbar(tr_cax, max([1e4, np.nanmax(self.temperature)]),
                            cmin=100., position='right',
                            orientation='vertical', numlevels=50,
                            colmap='plasma', norm=im_T.norm)
        tr_cax.set_ylim(100., 1e4)

        im_xi = bl_ax.imshow(self.ion_fraction[self.ny // 2].T * 100., vmin=0.,
                             vmax=100.0,
                             extent=(np.min(self.grid[1]),
                                     np.max(self.grid[1]) + self.csize * 1.,
                                     np.min(self.grid[2]),
                                     np.max(self.grid[2]) + self.csize * 1.),
                             cmap='gnuplot', aspect="equal")
        bl_ax.set_xlim(np.array(bl_ax.get_ylim()) * aspect)
        pfunc.make_colorbar(bl_cax, 100., cmin=0., position='right',
                            orientation='vertical', numlevels=50,
                            colmap='gnuplot', norm=im_xi.norm)
        bl_cax.set_yticks(np.linspace(0., 100., 6))

        im_vs = br_ax.imshow(self.vel[1][self.ny // 2].T,
                             vmin=np.nanmin(self.vel[1]),
                             vmax=np.nanmax(self.vel[1]),
                             extent=(np.min(self.grid[1]),
                                     np.max(self.grid[1]) + self.csize * 1.,
                                     np.min(self.grid[2]),
                                     np.max(self.grid[2]) + self.csize * 1.),
                             cmap='coolwarm', aspect="equal")
        br_ax.set_xlim(np.array(br_ax.get_ylim()) * aspect)
        pfunc.make_colorbar(br_cax, np.nanmax(self.vel[1]),
                            cmin=np.nanmin(self.vel[1]), position='right',
                            orientation='vertical', numlevels=50,
                            colmap='coolwarm', norm=im_vs.norm)

        dx = int((np.ptp(br_ax.get_xlim()) / self.csize) // 2 * 2 // 20)
        dz = self.nz // 10
        vzs = self.vel[2][self.ny // 2][::dx, ::dz].flatten()
        xs = self.grid[0][self.ny // 2][::dx, ::dz].flatten()[~np.isnan(vzs)]
        zs = self.grid[2][self.ny // 2][::dx, ::dz].flatten()[~np.isnan(vzs)]
        vzs = vzs[~np.isnan(vzs)]
        cs = br_ax.transAxes.transform((0.15, 0.5))
        cs = br_ax.transData.inverted().transform(cs)

        v_scale = np.ceil(np.max(vzs) / 10 ** np.floor(np.log10(np.max(vzs))))
        v_scale *= 10 ** np.floor(np.log10(np.max(vzs)))

        # Max arrow length is 0.1 * the height of the subplot
        scale = v_scale * 0.1 ** -1.
        br_ax.quiver(xs, zs, np.zeros((len(xs),)), vzs,
                     color='w', scale=scale,
                     scale_units='height')

        br_ax.quiver(cs[0], cs[1], [0.], [v_scale], color='k', scale=scale,
                     scale_units='height', pivot='tail')

        br_ax.annotate(r'$' + format(v_scale, '.0f') + '$\n$' +
                       r'\rm{km/s}$', cs, xytext=(0., -5.),  # half fontsize
                       xycoords='data', textcoords='offset points', va='top',
                       ha='center', multialignment='center', fontsize=10)

        axes = [tl_ax, tr_ax, bl_ax, br_ax]
        caxes = [tl_cax, tr_cax, bl_cax, br_cax]

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
            xlims = ax.get_xlim()
            ax.set_xticks(ax.get_yticks())
            ax.set_xlim(xlims)
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
            plt.savefig(savefig, bbox_inches='tight', dpi=300)

        return None

    def radio_plot(self, freq, percentile=5., savefig=False):
        """
        Generate 3 subplots of (from left to right) flux, optical depth and
        emission measure.
        
        Parameters
        ----------
        freq : float,
            Frequency to produce images at.
            
        percentile : float,
            Percentile of pixels to exclude from colorscale. Implemented as
            some edge pixels have extremely low values. Supplied value must be
            between 0 and 100.

        savefig: bool, str
            Whether to save the radio plot to file. If False, will not, but if
            a str representing a valid path will save to that path.

        Returns
        -------
        None.
        """
        import matplotlib.gridspec as gridspec

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
        import matplotlib
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

        flux = self.flux_ff(freq) * 1e3
        taus = self.optical_depth_ff(freq)
        taus = np.where(taus > 0, taus, np.NaN)
        ems = self.emission_measure()
        ems = np.where(ems > 0., ems, np.NaN)

        csize_as = np.tan(self.csize * con.au / con.parsec /
                          self.params['target']['dist'])  # radians
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
        pfunc.make_colorbar(l_cax, np.nanmax(flux), cmin=flux_min,
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
        pfunc.make_colorbar(m_cax, np.nanmax(taus), cmin=tau_min,
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
        pfunc.make_colorbar(r_cax, np.nanmax(ems), cmin=em_min,
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
            plt.savefig(savefig, bbox_inches='tight', dpi=300)

        return None

    def jml_profile_plot(self, ax=None, savefig=False):
        """
        Plot ejection profile using matlplotlib5

        Parameters
        ----------
        ax : matplotlib.axes._axes.Axes
            Axis to plot to

        times : np.array of astropy.units.quantity.Quantity instances
            Times to calculate mass loss rates at
        Returns
        -------
        numpy.array giving mass loss rates

        """
        # Plot out to 5 half-lives away from last existing burst in profile
        t_0s = [self._ejections[_]['t_0'] for _ in self._ejections]
        hls = [self._ejections[_]['half_life'] for _ in self._ejections]
        t_max = np.max(np.array(t_0s + 5 * np.array(hls)))

        times = np.linspace(0, t_max, 1000)
        jmls = self._jml_t(times)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6.65, 6.65))

        ax.plot(times / con.year, jmls * con.year / 1.98847e30, ls='-',
                color='cornflowerblue')

        xunit = u.format.latex.Latex(times).to_string(u.year)
        yunit = u.format.latex.Latex(jmls).to_string(u.solMass * u.year ** -1)

        xunit = r' \left[ ' + xunit.replace('$', '') + r'\right] $'
        yunit = r' \left[ ' + yunit.replace('$', '') + r'\right] $'

        ax.set_xlabel(r"$ t \," + xunit)
        ax.set_ylabel(r"$ \dot{m}_{\rm jet}\," + yunit)

        if savefig:
            plt.savefig(savefig, bbox_inches='tight', dpi=300)

        return None


class ModelRun:
    """
    Class to handle a creation of physical jet model, creation of .fits files
    and subsequent synthetic imaging via CASA.
    """

    @classmethod
    def load_pipeline(cls, load_file):
        """
        Loads pipeline from a previously saved state

        Parameters
        ----------
        cls : ModelRun
            DESCRIPTION.
        load_file : str
            Full path to saved ModelRun file (pickle file).

        Returns
        -------
        Instance of ModelRun initiated from save state.
        """
        loaded = pickle.load(open(load_file, 'rb'))

        # for idx, run in enumerate(loaded['runs']):
        #     for key in run:
        #         if type(run[key]) is str:
        #             if not os.path.exists(run['key']):
        #                 run[key] = run[key].replace('/Users/simon', '~')
        #         except AttributeError:
        #             pass
        #     loaded['runs'][idx] = run

        jm = JetModel.load_model(loaded["model_file"])
        params = loaded["params"]

        new_modelrun = cls(jm, params)
        new_modelrun.runs = loaded["runs"]

        return new_modelrun

    def __init__(self, jetmodel, params):
        """

        Parameters
        ----------
        jetmodel : JetModel
            Instance of JetModel to work with
        params : dict or str
            Either a dictionary containing all necessary radiative transfer and
            synthetic observation parameters, or a full path to a parameter
            file.
        """
        if isinstance(jetmodel, JetModel):
            self.model = jetmodel
        else:
            raise TypeError("Supplied arg jetmodel must be JetModel instance "
                            "not {}".format(type(jetmodel)))

        if isinstance(params, dict):
            self._params = params
        elif isinstance(params, str):
            if not os.path.exists(params):
                raise FileNotFoundError(params + " does not exist")
            if os.path.dirname(params) not in sys.path:
                sys.path.append(os.path.dirname(params))

            jp = __import__(os.path.basename(params)[:-3])
            self._params = jp.params
        else:
            raise TypeError("Supplied arg params must be dict or str")

        self.model_dcy = self.params['dcys']['model_dcy']
        self.model_file = self.model_dcy + os.sep + "jetmodel.save"
        self.save_file = self.model_dcy + os.sep + "modelrun.save"

        # Create Log for ModelRun instance
        log_name = "ModelRun_"
        log_name += time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
        log_name += ".log"

        self._dcy = self.params['dcys']['model_dcy']

        if not os.path.exists(self.dcy):
            os.mkdir(self.dcy)
            fn = os.sep.join([self.dcy, log_name])
            self._log = logger.Log(fname=fn)
            self._log.add_entry(mtype="INFO",
                                entry="Creating model directory, " +
                                      self.dcy)
        else:
            self._log = logger.Log(fname=os.sep.join([self.dcy, log_name]))

        self.model.log = self.log
        self.params['times'].sort()
        self.params['freqs'].sort()

        # Create directory names for all times here
        days = ["Day" + str(int(_ * 365.)) for _ in self.params['times']]
        freq_strs = [str(int(_ / 1e6)) + "MHz" for _ in self.params['freqs']]

        runs = []
        for idx1, day in enumerate(days):
            for idx2, freq in enumerate(freq_strs):
                rdcy = os.sep.join([self.dcy, day, freq])
                ff = rdcy + os.sep + '_'.join(['*', day, freq]) + '.fits'
                runs.append({'dcy': rdcy,
                             'fits_flux': ff.replace('*', 'Flux'),
                             'fits_tau': ff.replace('*', 'Tau'),
                             'fits_em': ff.replace('*', 'EM'),
                             'time': self.params['times'][idx1],
                             'freq': self.params['freqs'][idx2],
                             'tscop': self.params['tscps'][idx2],
                             't_int': int(self.params['t_ints'][idx2]),
                             't_obs': int(self.params['t_obs'][idx2]),
                             'bw': self.params['bws'][idx2],
                             'nchan': int(self.params['nchans'][idx2]),
                             'completed': False})
        self._runs = runs

    def save(self, save_file):
        p = {"runs": self.runs,
             "params": self._params,
             "model_file": self.model_file}
        pickle.dump(p, open(save_file, 'wb'))
        return None

    @property
    def params(self):
        return self._params

    @property
    def dcy(self):
        return self._dcy

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, new_model):
        self._model = new_model

    @property
    def runs(self):
        return self._runs

    @runs.setter
    def runs(self, new_runs):
        self._runs = new_runs

    @property
    def log(self):
        return self._log

    def execute(self, simobserve=True, verbose=True, dryrun=False,
                resume=True, clobber=False):
        """
        Execute a complete set of runs for the model to produce model data
        files, .fits files, CASA's simobserve's measurement sets
        and/or CASA's clean task's imaging products.

        Parameters
        ----------
        simobserve: bool
            Whether to produce synthetic visibilities for the designated runs
            using CASA and produce CLEAN images
        verbose: bool
            Verbose output in the terminal?
        dryrun: bool
            Whether to dry run, just producing set of commands to terminal
        resume: bool
            Whether to resume a previous run, if self.model.model_file and
            self.save_file exists. Default is True.
        clobber: bool
            Whether to redo 'completed' runs or not. Default is False

        Returns
        -------
        Dictionary of data products from each part of the execution
        """
        if verbose != self.log.verbose:
            self.log.verbose = verbose

        # Target coordinates as SkyCoord instance
        tgt_c = SkyCoord(self.model.params['target']['ra'],
                         self.model.params['target']['dec'],
                         unit=(u.hourangle, u.degree), frame='fk5')

        if simobserve:
            from datetime import datetime, timedelta
            import RaJePy.casa as casa
            import RaJePy.casa.tasks as tasks
            import RaJePy.maths as maths

            # Make pointing file
            ptgfile = self.model_dcy + os.sep + 'pointings.ptg'
            ptg_txt = "#Epoch     RA          DEC      TIME(optional)\n"
            ptg_txt += "J2000 {} ".format(tgt_c.to_string('hmsdms'))

            with open(ptgfile, 'wt') as f:
                f.write(ptg_txt)

        if resume:
            if os.path.exists(self.model_file):
                self.model = JetModel.load_model(self.model_file)

        for idx, run in enumerate(self.runs):
            if run["completed"] and not clobber:
                continue
            try:
                # Create relevant directories
                if not os.path.exists(run['dcy']):
                    self.log.add_entry(mtype="INFO",
                                       entry="Creating directory, " +
                                             run['dcy'])
                    # if not dryrun:
                    os.makedirs(run['dcy'])

                if "products" not in self.runs[idx].keys():
                    self.runs[idx]["products"] = {}

                t_str = os.path.basename(os.path.dirname(run['dcy']))
                freq_str = os.path.basename(run['dcy'])

                ff = run['dcy'] + os.sep
                ff += '_'.join(['*', t_str, freq_str]) + '.fits'

                run['fits_flux'] = ff.replace('*', 'Flux')

                # Plot physical jet model if required
                model_plot = os.sep.join([os.path.dirname(run['dcy']),
                                          "ModelPlot.pdf"])

                if not dryrun:
                    if not os.path.exists(model_plot):
                        self.model.model_plot(savefig=model_plot)

                    # Compute Emission measures for model plots
                    self.model.emission_measure(savefits=run['fits_em'])

                    # Run radiative transfer
                    self.model.optical_depth_ff(run['freq'],
                                                savefits=run['fits_tau'])
                    flux_vals = self.model.flux_ff(run['freq'],
                                                   savefits=run['fits_flux'])

                    self.radio_plot(run, savefig=os.sep.join([run['dcy'],
                                                              "RadioPlot.pdf"]))
                    self.runs[idx]['flux'] = np.nansum(flux_vals)

                    # Save model data if doesn't exist
                    if not os.path.exists(self.model_file):
                        self.model.save(self.model_file)

                    # Save pipeline state after successful run
                    self.save(self.save_file)

            except KeyboardInterrupt:
                self.save(self.save_file)
                self.model.save(self.model_file)
                raise KeyboardInterrupt("Pipeline interrupted by user")

            # Run casa's simobserve and produce visibilities, followed by tclean
            # and then export the images in .fits format
            if simobserve:
                script = casa.Script()
                os.chdir(run['dcy'])

                # Get desired telescope name
                tscop = run['tscop'][0]

                # Get antennae positions file's path
                t_cfg = run['tscop'][1]
                ant_list = casa.observatories.cfg_files[tscop][t_cfg]

                # Set frequency and bandwidth strings
                if run['freq'] < 10e9:
                    freq_str = '{:.0f}MHz'.format(run['freq'] / 1e6)

                else:
                    freq_str = '{:.0f}GHz'.format(run['freq'] / 1e9)

                if run['bw'] < 1e9:
                    bw_str = '{:.0f}MHz'.format(run['bw'] / 1e6)

                else:
                    bw_str = '{:.0f}GHz'.format(run['bw'] / 1e9)

                # Get hour-angle ranges above minimum elevation
                min_el = self.params['min_el']
                tscop_lat = casa.observatories.Lat[tscop]

                min_ha = tgt_c.ra.hour - 12.
                if min_ha < 0: min_ha += 24.

                el_range = (maths.astronomy.elevation(tgt_c, tscop_lat,
                                                      min_ha),
                            maths.astronomy.elevation(tgt_c, tscop_lat,
                                                      tgt_c.ra.hour))

                # Time above elevation limit in seconds, per day
                if min(el_range) > min_el:
                    time_up = int(24. * 60. * 60.)
                else:
                    time_up = 7200. * maths.astronomy.ha(tgt_c, tscop_lat,
                                                         min_el)
                    time_up = int(time_up)

                # Determine if multiple measurement sets are required (e.g. for
                # E-W interferometers, or other sparsely-filled snapshot
                # apertures, or elevation limits imposing observeable times
                # too short for desired time on target per day)
                multiple_ms = False  # Are multiple `observational runs' rqd?
                ew_int = False  # Is the telescope an E-W interferometer?
                nruns = 1  # Number of `observational runs'

                # Number of scans through ha-range for E-W interferometers
                # during the final `day of observations' --> ARBITRARY
                # HARD-CODED VALUE SET OF 8 SCANS
                ew_split_final_n = 8

                if tscop in ('ATCA', 'WSRT'):
                    ew_int = True

                if ew_int or time_up < run['t_obs']:
                    multiple_ms = True

                totaltimes = [time_up] * (run['t_obs'] // time_up)
                totaltimes += [run['t_obs'] - run['t_obs'] // time_up * time_up]

                # Decide 'dates of observation'
                refdates = []
                for n in range(len(totaltimes)):
                    rdate = (datetime.now() + timedelta(days=n))
                    rdate = rdate.strftime("%Y/%m/%d")
                    refdates.append(rdate)

                # Central hour angles for each observation
                hourangles = ['0h'] * len(totaltimes)

                if ew_int:
                    hourangles.pop(-1)
                    final_refdate = refdates.pop(-1)
                    final_t_obs = totaltimes.pop(-1)
                    total_gap = time_up - final_t_obs
                    t_gap = int(total_gap / (ew_split_final_n - 1))
                    t_scan = int(final_t_obs / ew_split_final_n)
                    for n in range(1, ew_split_final_n + 1):
                        ha = -time_up / 2 + t_scan / 2 + \
                             (t_gap + t_scan) * (n - 1)
                        hourangles.append('{:.5f}h'.format(ha / 3600.))
                        refdates.append(final_refdate)
                        totaltimes.append(t_scan)

                projects = ['SynObs' + str(n) for n in range(len(totaltimes))]

                if not multiple_ms:
                    projects = ['SynObs']

                # Synthetic observations
                for idx2, totaltime in enumerate(totaltimes):
                    so = tasks.Simobserve(project=projects[idx2],
                                          skymodel=run['fits_flux'],
                                          incenter=freq_str,
                                          inwidth=bw_str,
                                          setpointings=False,
                                          ptgfile=ptgfile,
                                          integration=str(run['t_int']) + 's',
                                          antennalist=ant_list,
                                          refdate=refdates[idx2],
                                          hourangle=hourangles[idx2],
                                          totaltime=str(totaltime) + 's',
                                          graphics='none',
                                          overwrite=True,
                                          verbose=True)
                    script.add_task(so)

                # Final measurement set paths
                fnl_clean_ms = run['dcy'] + os.sep + 'SynObs' + os.sep
                fnl_clean_ms += '.'.join(['SynObs',
                                          os.path.basename(ant_list).strip(
                                              '.cfg'),
                                          'ms'])

                fnl_noisy_ms = run['dcy'] + os.sep + 'SynObs' + os.sep
                fnl_noisy_ms += '.'.join(['SynObs',
                                          os.path.basename(ant_list).strip(
                                              '.cfg'),
                                          'noisy', 'ms'])

                if multiple_ms:
                    script.add_task(tasks.Mkdir(name='SynObs'))
                    clean_mss, noisy_mss = [], []

                    for project in projects:
                        pdcy = run['dcy'] + os.sep + project
                        clean_ms = '.'.join([project,
                                             os.path.basename(
                                                 ant_list).strip('.cfg'),
                                             'ms'])
                        noisy_ms = '.'.join([project,
                                             os.path.basename(
                                                 ant_list).strip('.cfg'),
                                             'noisy', 'ms'])
                        clean_mss.append(pdcy + os.sep + clean_ms)
                        noisy_mss.append(pdcy + os.sep + noisy_ms)

                    script.add_task(tasks.Concat(vis=clean_mss,
                                                 concatvis=fnl_clean_ms))

                    script.add_task(tasks.Concat(vis=noisy_mss,
                                                 concatvis=fnl_noisy_ms))
                    for project in projects:
                        pdcy = run['dcy'] + os.sep + project
                        script.add_task(tasks.Rmdir(path=pdcy))

                script.add_task(tasks.Chdir(run['dcy'] + os.sep + 'SynObs'))

                # Determine spatial resolution and hence cell size
                ant_data = {}
                with open(ant_list, 'rt') as f:
                    for i, line in enumerate(f.readlines()):
                        if line[0] != '#':
                            line = line.split()
                            ant_data[line[4]] = [float(_) for _ in line[:3]]

                ants = list(ant_data.keys())

                ant_pairs = {}
                for i, ant1 in enumerate(ants):
                    if i != len(ants) - 1:
                        for ant2 in ants[i + 1:]:
                            ant_pairs[(ant1, ant2)] = np.sqrt(
                                (ant_data[ant1][0] - ant_data[ant2][0]) ** 2 +
                                (ant_data[ant1][1] - ant_data[ant2][1]) ** 2 +
                                (ant_data[ant1][2] - ant_data[ant2][2]) ** 2)

                max_bl_uvwave = max(ant_pairs.values()) / (con.c / run['freq'])
                beam_min = 1. / max_bl_uvwave / con.arcsec

                cell_str = '{:.6f}arcsec'.format(beam_min / 4.)
                cell_size = float('{:.6f}'.format(beam_min / 4.))

                # Define cleaning region as the box encapsulating the flux-model
                # and determine minimum clean image size as twice that of the
                # angular coverage of the flux-model
                ff = fits.open(run['fits_flux'])[0]
                fm_head = ff.header

                nx, ny = fm_head['NAXIS1'], fm_head['NAXIS2']
                cpx, cpy = fm_head['CRPIX1'], fm_head['CRPIX2']
                cx, cy = fm_head['CRVAL1'], fm_head['CRVAL2']
                cellx, celly = fm_head['CDELT1'], fm_head['CDELT2']

                blc = (cx - cellx * cpx, cy - celly * cpy)
                trc = (blc[0] + cellx * nx, blc[1] + celly * ny)

                # Get peak flux expected from observations for IMFIT task later
                fm_data = ff.data

                # Just get 2-D intensity data from RA/DEC axes
                while len(np.shape(fm_data)) > 2:
                    fm_data = fm_data[0]

                # Create arcsec-offset coordinate grid of .fits model data
                # relative to central pixel
                xx, yy = np.meshgrid(np.arange(nx) + 0.5 - cpx,
                                     np.arange(ny) + 0.5 - cpy)

                xx *= cellx * 3600.  # to offset-x (arcsecs)
                yy *= celly * 3600.  # to offset-y (arcsecs)
                rr = np.sqrt(xx ** 2. + yy ** 2.)  # Distance from jet-origin

                peak_flux = np.nansum(np.where(rr < beam_min / 2., fm_data, 0.))

                # Derive jet major and minor axes from tau = 1 surface
                r_0_au = self.model.params['geometry']['r_0']
                w_0_au = self.model.params['geometry']['w_0']
                tau_0 = mphys.tau_r(self.model, run['freq'], r_0_au)

                r_0_as = np.arctan(r_0_au * con.au / con.parsec /
                                   self.model.params['target']['dist'])  # rad
                r_0_as /= con.arcsec  # arcsec

                w_0_as = np.arctan(w_0_au * con.au / con.parsec /
                                   self.model.params['target']['dist'])  # rad
                w_0_as /= con.arcsec  # arcsec

                q_tau = self.model.params['power_laws']['q_tau']
                eps = self.model.params['geometry']['epsilon']
                jet_deconv_maj = 2. * r_0_as * (1. / tau_0)**(1. / q_tau)
                jet_deconv_min = 2. * w_0_as * (jet_deconv_maj / 2. /
                                                r_0_as)**eps

                jet_conv_maj = np.sqrt(jet_deconv_maj**2. + beam_min**2.)
                jet_conv_min = np.sqrt(jet_deconv_min**2. + beam_min**2.)

                mask_str = 'box[[{}deg, {}deg], [{}deg, {}deg]]'.format(blc[0],
                                                                        blc[1],
                                                                        trc[0],
                                                                        trc[1])

                min_imsize_as = max(np.abs([nx * cellx, ny * celly])) * 7200.
                min_imsize_cells = int(np.ceil(min_imsize_as / cell_size))

                if min_imsize_cells < 500:
                    imsize_cells = [500, 500]
                else:
                    imsize_cells = [min_imsize_cells] * 2

                im_name = fnl_noisy_ms.strip('ms') + 'imaging'

                # Deconvolution of final, noisy, synthetic dataset
                script.add_task(tasks.Tclean(vis=fnl_noisy_ms,
                                             imagename=im_name,
                                             imsize=imsize_cells,
                                             cell=[cell_str],
                                             weighting='briggs',
                                             robust=0.5,
                                             niter=500,
                                             nsigma=2.5,
                                             mask=mask_str,
                                             interactive=False))

                fitsfile = run['dcy'] + os.sep + os.path.basename(im_name)
                fitsfile += '.fits'

                script.add_task(tasks.Exportfits(imagename=im_name + '.image',
                                                 fitsimage=fitsfile))

                imfit_estimates_file = fitsfile.replace('fits', 'estimates')

                est_str = '{:.6f}, {:.1f}, {:.1f}, {:.5f}arcsec, ' \
                          '{:.5f}arcsec, ' \
                          '{:.2f}deg'
                est_str = est_str.format(peak_flux, imsize_cells[0] / 2.,
                                         imsize_cells[1] / 2.,
                                         jet_conv_maj, jet_conv_min,
                                         self.model.params['geometry']['pa'])

                with open(imfit_estimates_file, 'wt') as f:
                    f.write(est_str)
                imfit_results = fitsfile.replace('fits', 'imfit')
                script.add_task(tasks.Imfit(imagename=fitsfile,
                                            estimates=imfit_estimates_file,
                                            summary=imfit_results))
                script.execute(dcy=run['dcy'], dryrun=dryrun)

                run['imfit'] = {}
                with open(imfit_results, 'rt') as f:
                    for idx3, line in enumerate(f.readlines()):
                        if idx3 == 0:
                            units = [''] + line[1:].split()
                        elif idx3 == 1:
                            h = line[1:].split()
                        else:
                            line = [float(_) for _ in line.split()]
                    for idx4, val in enumerate(line):
                        run['imfit'][h[idx4]] = {'val': val,
                                                 'unit': units[idx4]}

                run['ms_noisy'] = fnl_noisy_ms
                run['ms_clean'] = fnl_clean_ms
                run['clean_image'] = fitsfile

            self.runs[idx]['completed'] = True

        self.save(self.save_file)
        self.model.save(self.model_file)

        return None  # self.runs[idx]['products']

    def plot_fluxes(self, plot_time):
        freqs, fluxes = [], []
        for idx, run in enumerate(self.runs):
            if run['time'] == plot_time:
                fluxes.append(np.nansum(fits.open(run['fits_flux'])[0].data[0]))
                freqs.append(run['freq'])

        freqs = np.array(freqs)
        fluxes = np.array(fluxes)

        alphas = []
        for n in np.arange(1, len(fluxes)):
            alphas.append(np.log10(fluxes[n] /
                                   fluxes[n - 1]) /
                          np.log10(freqs[n] / freqs[n - 1]))
        alphas = np.append(alphas[0], alphas)

        plt.close('all')

        fig, ax1 = plt.subplots(1, 1, figsize=(6., 6.))
        ax2 = ax1.twinx()

        ax2.plot(freqs, alphas, color='b', ls='None', mec='b', marker='o',
                 mfc='cornflowerblue', lw=2, zorder=2)

        freqs_r86 = np.logspace(np.log10(np.min(freqs)),
                                np.log10(np.max(freqs)), 100)
        flux_exp = []
        for freq in freqs_r86:
            if self.model.params['grid']['l_z'] is not None:
                f = mphys.flux_expected_r86(self.model, freq,
                                            self.model.params['grid']['l_z'] /
                                            2)
            else:
                scale = np.arctan(self.model.params['grid']['n_z'] *
                                  self.model.params['grid']['c_size'] * 0.5 /
                                  (self.model.params['target']['dist'] *
                                   con.parsec / con.au)) / con.arcsec
                f = mphys.flux_expected_r86(self.model, freq, scale)
            flux_exp.append(f * 2.)  # for biconical jet

        alphas_r86 = []
        for n in np.arange(1, len(freqs_r86)):
            alphas_r86.append(np.log10(flux_exp[n] / flux_exp[n - 1]) /
                              np.log10(freqs_r86[n] / freqs_r86[n - 1]))
        alphas_r86 = np.append(alphas_r86[0], alphas_r86)

        ax2.plot(freqs_r86, alphas_r86, color='cornflowerblue', ls='--', lw=2,
                 zorder=1)

        ax1.loglog(freqs, fluxes, mec='maroon', ls='None', mfc='r', lw=2,
                   zorder=3, marker='o')
        ax1.loglog(freqs_r86, flux_exp, color='gray', ls='-', lw=2, zorder=1)
        ax1.loglog(freqs_r86,
                   mphys.approx_flux_expected_r86(self.model, freqs_r86) * 2.,
                   color='gray', ls='-.', lw=2, zorder=1)
        ax1.set_xlim(np.min(freqs), np.max(freqs))
        pfunc.equalise_axes(ax1)

        ax1.set_xlabel(r'$\nu \, \left[ {\rm Hz} \right]$', color='k')
        ax1.set_ylabel(r'$S_\nu \, \left[ {\rm Jy} \right]$', color='k')
        ax2.set_ylabel(r'$\alpha$', color='b')

        ax1.tick_params(which='both', direction='in', top=True)
        ax2.tick_params(which='both', direction='in', color='b')
        ax2.tick_params(axis='y', which='both', colors='b')
        ax2.spines['right'].set_color('b')
        ax2.yaxis.label.set_color('b')

        save_file = '_'.join(
            ["Jet", "lz" + str(self.model.params["grid"]["l_z"]),
             "csize" + str(self.model.params["grid"]["c_size"])])
        save_file += '.png'

        fig.savefig('/Users/simon/Desktop/' + save_file, bbox_inches='tight')

        return None

    def radio_plot(self, run, percentile=5., savefig=False):
        """
        Generate 3 subplots of (from left to right) flux, optical depth and
        emission measure.
        
        Parameters
        ----------
        run : dict,
            One of the ModelRun instance's runs
            
        percentile : float,
            Percentile of pixels to exclude from colorscale. Implemented as
            some edge pixels have extremely low values. Supplied value must be
            between 0 and 100.

        savefig: bool, str
            Whether to save the radio plot to file. If False, will not, but if
            a str representing a valid path will save to that path.
    
        Returns
        -------
        None.
        """
        import matplotlib.gridspec as gridspec

        freq = run['freq']

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

        flux = fits.open(run['fits_flux'])[0].data[0]
        taus = fits.open(run['fits_tau'])[0].data[0]
        ems = fits.open(run['fits_em'])[0].data[0]

        flux = np.where(flux > 0, flux, np.NaN)
        taus = np.where(taus > 0, taus, np.NaN)
        ems = np.where(ems > 0, ems, np.NaN)

        csize_as = np.tan(self.model.csize * con.au / con.parsec /
                          self.model.params['target']['dist'])  # radians
        csize_as /= con.arcsec  # arcseconds
        x_extent = np.shape(flux)[1] * csize_as
        z_extent = np.shape(flux)[0] * csize_as

        flux_min = np.nanpercentile(flux, percentile)
        if np.log10(flux_min) > (np.log10(np.nanmax(flux)) - 1.):
            flux_min = 10 ** (np.floor(np.log10(np.nanmax(flux)) - 1.))

        im_flux = l_ax.imshow(flux,
                              norm=LogNorm(vmin=flux_min,
                                           vmax=np.nanmax(flux)),
                              extent=(-x_extent / 2., x_extent / 2.,
                                      -z_extent / 2., z_extent / 2.),
                              cmap='gnuplot2_r', aspect="equal")

        l_ax.set_xlim(np.array(l_ax.get_ylim()) * aspect)
        pfunc.make_colorbar(l_cax, np.nanmax(flux), cmin=flux_min,
                            position='right', orientation='vertical',
                            numlevels=50, colmap='gnuplot2_r',
                            norm=im_flux.norm)

        tau_min = np.nanpercentile(taus, percentile)
        im_tau = m_ax.imshow(taus,
                             norm=LogNorm(vmin=tau_min,
                                          vmax=np.nanmax(taus)),
                             extent=(-x_extent / 2., x_extent / 2.,
                                     -z_extent / 2., z_extent / 2.),
                             cmap='Blues', aspect="equal")
        m_ax.set_xlim(np.array(m_ax.get_ylim()) * aspect)
        pfunc.make_colorbar(m_cax, np.nanmax(taus), cmin=tau_min,
                            position='right', orientation='vertical',
                            numlevels=50, colmap='Blues',
                            norm=im_tau.norm)

        em_min = np.nanpercentile(ems, percentile)
        im_EM = r_ax.imshow(ems,
                            norm=LogNorm(vmin=em_min,
                                         vmax=np.nanmax(ems)),
                            extent=(-x_extent / 2., x_extent / 2.,
                                    -z_extent / 2., z_extent / 2.),
                            cmap='cividis', aspect="equal")
        r_ax.set_xlim(np.array(r_ax.get_ylim()) * aspect)
        pfunc.make_colorbar(r_cax, np.nanmax(ems), cmin=em_min,
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
                                   np.shape(flux)[1]),
                       np.linspace(-z_extent / 2., z_extent / 2.,
                                   np.shape(flux)[0]),
                       taus, [1.], colors='w')
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
            plt.savefig(savefig, bbox_inches='tight', dpi=300)

        return None


class Pointing(object):
    """
    Class to handle a single pointing and all of its information
    """

    def __init__(self, time, ra, dec, duration, epoch='J2000'):
        import astropy.units as u
        from astropy.coordinates import SkyCoord

        self._time = time
        self._duration = duration

        if epoch == 'J2000':
            frame = 'fk5'
        elif epoch == 'B1950':
            frame = 'fk4'
        else:
            raise ValueError("epoch, {}, is unsupported. Must be J2000 or "
                             "B1950".format(epoch))

        self._coord = SkyCoord(ra, dec, unit=(u.hour, u.deg), frame=frame)
        self._epoch = epoch

    @property
    def time(self):
        return self._time

    @property
    def ra(self):
        h = self.coord.ra.hms.h
        m = self.coord.ra.hms.m
        s = self.coord.ra.hms.s
        return '{:02.0f}h{:02.0f}m{:06.3f}'.format(h, m, s)

    @property
    def ra(self):
        d = self.coord.dec.dms.d
        m = self.coord.dec.dms.m
        s = self.coord.dec.dms.s
        return '{:+03.0f}d{:02.0f}m{:06.3f}'.format(d, m, s)

    @property
    def dec(self):
        return self._dec

    @property
    def duration(self):
        return self._duration

    @property
    def epoch(self):
        return self._epoch

    @property
    def coord(self):
        return self._coord


class PointingScheme(object):
    """
    Class to handle the pointing scheme for synthetic observations
    """

    def __init__(self):
        self.pointings = []


if __name__ == '__main__':
    from RaJePy.classes import JetModel, ModelRun
    import RaJePy as rjp

    jm = JetModel(rjp.cfg.dcys['files'] + os.sep + 'example-model-params.py')
    pline = ModelRun(jm, rjp.cfg.dcys['files'] + os.sep +
                     'example-pipeline-params.py')

    pline.execute(simobserve=True, dryrun=False)
