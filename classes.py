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
from typing import Union, Callable, List, Tuple, Dict

import numpy.typing as npt
import numpy as np
import tabulate
import astropy.units as u
import scipy.constants as con
from astropy.coordinates import SkyCoord
from astropy.io import fits
from matplotlib.colors import LogNorm

from RaJePy import logger
from RaJePy import _config as cfg
from RaJePy.maths import geometry as mgeom
from RaJePy.maths import physics as mphys
from RaJePy.maths import rrls as mrrl
from RaJePy.miscellaneous import functions as miscf
from RaJePy.plotting import functions as pfunc

from warnings import filterwarnings

filterwarnings("ignore", category=RuntimeWarning)


# noinspection PyCallingNonCallable
class JetModel:
    """
    Class to handle physical model of an ionised jet from a young stellar object
    """
    _arr_indexing = 'ij'  # numpy.meshgrid indexing type

    @classmethod
    def load_model(cls, model_file: str) -> "JetModel":
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
        model_file = os.path.expanduser(model_file)
        loaded = pickle.load(open(model_file, 'rb'))

        # Create new JetModel class instance
        if 'log' in loaded:
            new_jm = cls(loaded["params"], log=loaded['log'])
        else:
            dcy_ = os.path.expanduser('~')
            new_jm = cls(loaded["params"],
                         log=logger.Log(dcy_ + os.sep + 'temp.log'))

        # If fill factors/projected areas have been previously calculated,
        # assign to new instance
        if loaded['ffs'] is not None:
            new_jm.fill_factor = loaded['ffs']

        if loaded['areas'] is not None:
            new_jm.areas = loaded['areas']

        new_jm.time = loaded['time']

        return new_jm

    @staticmethod
    def lz_to_grid_dims(params: Dict) -> Tuple[int, int, int]:
        cs_au = params["grid"]["c_size"]
        i_rads = np.radians(params["geometry"]["inc"])
        pa_rads = np.radians(params["geometry"]["pa"])
        l_xz_au = params['grid']['l_z'] * params['target']['dist']

        xmax_au = l_xz_au * np.sin(pa_rads)
        ymax_au = l_xz_au * np.tan(1.571 - i_rads)
        zmax_au = l_xz_au * np.cos(pa_rads)

        rmax_au, _, __ = mgeom.xyz_to_rwp(xmax_au, ymax_au, zmax_au,
                                          params["geometry"]["inc"],
                                          params["geometry"]["pa"])
        wmax_au = mgeom.w_r(rmax_au,
                            params["geometry"]["w_0"],
                            params["geometry"]["mod_r_0"],
                            params["geometry"]["r_0"],
                            params["geometry"]["epsilon"])
        wmax_cells = int(np.ceil(np.abs(wmax_au / cs_au)))

        nx = int(np.ceil(np.abs(xmax_au / cs_au)))
        ny = int(np.ceil(np.abs(ymax_au / cs_au)))
        nz = int(np.ceil(np.abs(zmax_au / cs_au)))

        # Make sure jet's width within cell grid. Especially pertinent if
        # inclination or position angles are 0, 90, 180 or 270 deg
        nx, ny, nz = [n + 2 * wmax_cells for n in (nx, ny, nz)]

        # Enforce even number of cells in x, y and z dimensions
        nx, ny, nz = [_ if _ % 2 == 0 else _ + 1 for _ in (nx, ny, nz)]

        return nx, ny, nz

    @staticmethod
    def py_to_dict(py_file: str) -> Dict:
        """
        Convert .py file (full path as str) containing relevant model parameters
        to dict
        """
        if not os.path.exists(py_file):
            raise FileNotFoundError(py_file + " does not exist")
        if os.path.dirname(py_file) not in sys.path:
            sys.path.append(os.path.dirname(py_file))

        jp = __import__(os.path.basename(py_file).rstrip('.py'))
        err = miscf.check_model_params(jp.params)
        if err is not None:
            raise err

        sys.path.remove(os.path.dirname(py_file))

        return jp.params

    def __init__(self, params: Union[dict, str],
                 log: Union[None, logger.Log] = None):
        """

        Parameters
        ----------
        params : dict
            dictionary containing all necessary parameters to describe
            physical jet model
        log: logger.Log
            Log instance to handle all log messages
        """
        # Import jet parameters
        if isinstance(params, dict):
            self._params = params
        elif isinstance(params, str):
            self._params = JetModel.py_to_dict(params)
        else:
            raise TypeError("Supplied arg params must be dict or file path ("
                            "str)")

        self._name = self.params['target']['name']
        self._csize = self.params['grid']['c_size']

        # Automatically calculated parameters
        mr0 = mgeom.mod_r_0(self._params['geometry']['opang'],
                            self._params['geometry']['epsilon'],
                            self._params['geometry']['w_0'])
        q_n = mphys.q_n(self._params["geometry"]["epsilon"],
                        self._params["power_laws"]["q_v"])
        q_tau = mphys.q_tau(self._params["geometry"]["epsilon"],
                            self._params["power_laws"]["q_x"],
                            q_n,
                            self._params["power_laws"]["q_T"])
        self._params["geometry"]["mod_r_0"] = mr0
        self._params["power_laws"]["q_n"] = q_n
        self._params["power_laws"]["q_tau"] = q_tau

        if log is not None:
            self._log = log
        else:
            self._log = logger.Log(os.path.expanduser('~') + os.sep +
                                   'temp.log', verbose=True)

        # Determine number of cells in x, y, and z-directions
        if self.params['grid']['l_z'] is not None:
            nx, ny, nz = JetModel.lz_to_grid_dims(self.params)
            self.log.add_entry("INFO",
                               'For a (bipolar) jet length of {:.1f}", cell '
                               'size of {:.2f}au and distance of {:.0f}pc, a '
                               'grid size of (n_x, n_y, n_z) = ({}, {}, {}) '
                               'voxels is calculated'
                               ''.format(self.params['grid']['l_z'],
                                         self.params["grid"]["c_size"],
                                         self.params["target"]["dist"],
                                         nx, ny, nz))

        else:
            # Enforce even number of cells in every direction
            nx = (self.params['grid']['n_x'] + 1) // 2 * 2
            ny = (self.params['grid']['n_y'] + 1) // 2 * 2
            nz = (self.params['grid']['n_z'] + 1) // 2 * 2

        self.params['grid']['n_x'] = nx
        self.params['grid']['n_y'] = ny
        self.params['grid']['n_z'] = nz

        self._nx = nx  # number of cells in x
        self._ny = ny  # number of cells in y
        self._nz = nz  # number of cells in z

        # Create necessary class-instance attributes for all necessary grids
        self._ff = None  # cell fill factors
        self._areas = None  # cell projected areas along y-axis
        self._idxs = None  # Grid of cell indices
        self._grid = None  # grid of cell-centre positions
        self._rwp = None  # Grid of cells' centroids' r, w, p coordinates
        self._rreff = None  # grid of cell-centre r_eff-coordinates
        self._ts = None  # grid of cell-material times since launch
        self._m = None  # grid of cell-masses
        self._nd = None  # grid of cell number densities
        self._xi = None  # grid of cell ionisation fractions
        self._temp = None  # grid of cell temperatures
        self._v = None  # 3-tuple of cell x, y and z velocity components
        self._ss_jml = self.params["properties"]["mlr"] * 1.989e30 / con.year

        n_0 = mphys.n_0_from_mlr(self.params["properties"]["mlr"],
                                 self.params["properties"]["v_0"],
                                 self.params["geometry"]["w_0"],
                                 self.params["properties"]["mu"],
                                 self.params["power_laws"]["q^d_n"],
                                 self.params["power_laws"]["q^d_v"],
                                 self.params["target"]["R_1"],
                                 self.params["target"]["R_2"])
        self.params["properties"]["n_0"] = n_0

        # Create attribute for jet mass loss rate as a function of a time
        self._jml_t = lambda t: self._ss_jml  # JML as function of time function
        self._ejections = {}  # Record of any ejection events
        for idx, ejn_t0 in enumerate(self.params['ejection']['t_0']):
            self.add_ejection_event(ejn_t0 * con.year,
                                    self._ss_jml *
                                    self.params['ejection']['chi'][idx],
                                    self.params['ejection']['hl'][idx] *
                                    con.year)

        self._time = 0. * con.year  # Current time in jet model

    def __str__(self) -> str:
        p = self.params
        h = ['Parameter', 'Value']
        d = [('epsilon', format(p['geometry']['epsilon'], '+.3f')),
             ('opang', format(p['geometry']['opang'], '+.0f') + ' deg'),
             ('q_v', format(p['power_laws']['q_v'], '+.3f')),
             ('q_T', format(p['power_laws']['q_T'], '+.3f')),
             ('q_x', format(p['power_laws']['q_x'], '+.3f')),
             ('q_n', format(p['power_laws']['q_n'], '+.3f')),
             ('q^d_v', format(p['power_laws']['q^d_v'], '+.3f')),
             ('q^d_T', format(p['power_laws']['q^d_T'], '+.3f')),
             ('q^d_x', format(p['power_laws']['q^d_x'], '+.3f')),
             ('q^d_n', format(p['power_laws']['q^d_n'], '+.3f')),
             ('q_tau', format(p['power_laws']['q_tau'], '+.3f')),
             ('cell', format(p['grid']['c_size'], '.1f') + ' au'),
             ('w_0', format(p['geometry']['w_0'], '.2f') + ' au'),
             ('r_0', format(p['geometry']['r_0'], '.2f') + ' au'),
             ('v_0', format(p['properties']['v_0'], '.0f') + ' km/s'),
             ('x_0', format(p['properties']['x_0'], '.3f')),
             ('n_0', format(p['properties']['n_0'], '.3e') + ' cm^-3'),
             ('T_0', format(p['properties']['T_0'], '.0e') + ' K'),
             ('i', format(p['geometry']['inc'], '+.1f') + ' deg'),
             ('theta', format(p['geometry']['pa'], '+.1f') + ' deg'),
             ('D', format(p['target']['dist'], '+.0f') + ' pc'),
             ('M*', format(p['target']['M_star'], '+.1f') + ' Msol'),
             ('R_1', format(p['target']['R_1'], '+.1f') + ' au'),
             ('R_2', format(p['target']['R_2'], '+.1f') + ' au')]

        # Add current model time if relevant (i.e. bursts are included)
        if len(p['ejection']['t_0']) > 0:
            d.append(('t_now', format(self.time / con.year, '+.3f') + ' yr'))

        col1_width = max(map(len, [h[0]] + list(list(zip(*d))[0]))) + 2
        col2_width = max(map(len, [h[1]] + list(list(zip(*d))[1]))) + 2
        tab_width = col1_width + col2_width + 3

        hline = tab_width * '-'
        delim = '|'

        s = hline + '\n'
        s += '/' + format('JET MODEL', '^' + str(tab_width - 2)) + '/\n'
        s += hline + '\n'
        s += delim + delim.join([format(h[0], '^' + str(col1_width)),
                                 format(h[1], '^' + str(col2_width))]) + delim
        s += '\n' + hline + '\n'
        for line_ in d:
            s += delim + \
                 delim.join([format(line_[0], '^' + str(col1_width)),
                             format(line_[1], '^' + str(col2_width))]) + \
                 delim + '\n'
        s += hline + '\n'

        # Burst information below
        hb = ['t_0', 'FWHM', 'chi']
        units = ['[yr]', '[yr]', '']
        db = []
        for idx, t in enumerate(p["ejection"]["t_0"]):
            db.append((format(t, '.2f'),
                       format(p["ejection"]["hl"][idx], '.2f'),
                       format(p["ejection"]["chi"][idx], '.2f')))
        s += '/' + format('BURSTS', '^' + str(tab_width - 2)) + '/\n'
        s += hline + '\n'

        if len(db) == 0:
            s += delim + format(' None ',
                                '-^' + str(tab_width - 2)) + delim + '\n'
            s += hline + '\n'
            return s

        bcol1_w = bcol2_w = bcol3_w = (tab_width - 4) // 3

        if (tab_width - 4) % 3 > 0:
            bcol1_w += 1
            if (tab_width - 4) % 3 == 2:
                bcol2_w += 1

        # Burst header and units
        for line_ in (hb, units):
            s += delim + delim.join([format(line_[0], '^' + str(bcol1_w)),
                                     format(line_[1], '^' + str(bcol2_w)),
                                     format(line_[2], '^' + str(bcol3_w))]) + \
                 delim + '\n'
        s += hline + '\n'

        # Burst(s) information
        for line_ in db:
            s += delim + delim.join([format(line_[0], '^' + str(bcol1_w)),
                                     format(line_[1], '^' + str(bcol2_w)),
                                     format(line_[2], '^' + str(bcol3_w))]) + \
                 delim + '\n'
        s += hline + '\n'

        return s

    @property
    def los_axis(self) -> int:
        """Which numpy axis lies parallel to the observer's line-of-sight"""
        if self._arr_indexing == 'ij':
            return 1
        elif self._arr_indexing == 'xy':
            return 0
        else:
            raise ValueError("Unknown numpy array indexing "
                             f"({self._arr_indexing})")

    @property
    def time(self) -> float:
        """Model time in seconds"""
        return self._time

    @time.setter
    def time(self, new_time: float):
        self._time = new_time

    @property
    def jml_t(self) -> Callable[[Union[float, np.ndarray]],
                                Union[float, np.ndarray]]:
        """Callable for jet-mass loss rate as a function of time, which is
        the callable's sole arg. [kg/s]"""
        return self._jml_t

    @jml_t.setter
    def jml_t(self, new_jml_t: Callable[[Union[float, np.ndarray]],
                                        Union[float, np.ndarray]]):
        self._jml_t = new_jml_t

    def add_ejection_event(self, t_0: float, peak_jml: float, half_life: float):
        """
        Add ejection event in the form of a Gaussian ejection profile as a
        function of time

        Parameters
        ----------
        t_0
            Time of peak mass loss rate [s]
        peak_jml
            Highest jet mass loss rate of ejection burst [kg/s]
        half_life
            Time for mass loss rate to halve during the burst [s]

        Returns
        -------
        None.

        """

        def func(fnc: Callable[[float], float], _t_0: float, _peak_jml: float,
                 _half_life: float) -> Callable[[float], float]:
            """

            Parameters
            ----------
            fnc : Time dependent function giving current jet mass loss rate
            _t_0 : Time of peak of burst
            _peak_jml : Peak of burst's jet mass loss rate
            _half_life : FWHM of burst

            Returns
            -------
            Factory function returning function describing new time dependent
            mass loss rate incorporating input burst

            """

            def func2(t: float) -> float:
                """Gaussian profiled ejection event"""
                amp = _peak_jml - self._ss_jml
                sigma = _half_life * 2. / (2. * np.sqrt(2. * np.log(2.)))
                return fnc(t) + amp * np.exp(-(t - _t_0) ** 2. /
                                             (2. * sigma ** 2.))

            return func2

        self._jml_t = func(self._jml_t, t_0, peak_jml, half_life)

        record = {'t_0': t_0, 'peak_jml': peak_jml, 'half_life': half_life}
        self._ejections[str(len(self._ejections) + 1)] = record

    @property
    def indices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._idxs:
            return self._idxs
        self._idxs = tuple(np.meshgrid(np.arange(self.nx),
                                       np.arange(self.ny),
                                       np.arange(self.nz),
                                       indexing=self._arr_indexing))

        return self._idxs

    @property
    def ix(self) -> np.ndarray:
        return self.indices[0]

    @property
    def iy(self) -> np.ndarray:
        return self.indices[1]

    @property
    def iz(self) -> np.ndarray:
        return self.indices[2]

    @property
    def grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Array of cell grid coordinates (in au) of shape (nx, ny, nz).
        Coordinates are of the bottom, left, front cell corners in au.
        """
        if self._grid:
            return self._grid

        self._grid = (self.csize * (self.ix - self.nx // 2),
                      self.csize * (self.iy - self.ny // 2),
                      self.csize * (self.iz - self.nz // 2))

        return self._grid

    @property
    def xx(self) -> np.ndarray:
        return self.grid[0]

    @property
    def yy(self) -> np.ndarray:
        return self.grid[1]

    @property
    def zz(self) -> np.ndarray:
        return self.grid[2]

    @property
    def grid_rwp(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Grid of cells' centroids' r, w, p coordinates in au"""
        if self._rwp:
            return self._rwp

        self._rwp = mgeom.xyz_to_rwp(self.xx + self.csize / 2.,
                                     self.yy + self.csize / 2.,
                                     self.zz + self.csize / 2.,
                                     self.params["geometry"]["inc"],
                                     self.params["geometry"]["pa"])
        return self._rwp

    @property
    def rr(self) -> np.ndarray:
        """Grid of cells' centroids' r coordinates in au"""
        return self.grid_rwp[0]

    @property
    def ww(self) -> np.ndarray:
        """Grid of cells' centroids' w coordinates in au"""
        return self.grid_rwp[1]

    @property
    def pp(self) -> np.ndarray:
        """Grid of cells' centroids' phi coordinates in radians"""
        return self.grid_rwp[2]

    @property
    def rreff(self) -> np.ndarray:
        """Grid of cells' centroids' effective accretion disc radii in au"""
        if self._rreff is not None:
            return self._rreff

        self._rreff = mgeom.r_eff(self.ww, self.params["target"]["R_1"],
                                  self.params["target"]["R_2"],
                                  self.params['geometry']['w_0'],
                                  np.abs(self.rr),
                                  self.params['geometry']['mod_r_0'],
                                  self.params['geometry']['r_0'],
                                  self.params["geometry"]["epsilon"])

        return self._rreff

    @property
    def xs(self) -> np.ndarray:
        return self.grid[0][0][::, 0]

    @property
    def ys(self) -> np.ndarray:
        return self.grid[1][::, 0][::, 0]

    @property
    def zs(self) -> np.ndarray:
        return self.grid[2][0][0]

    @property
    def fill_factor(self) -> np.ndarray:
        """
        Calculate the fraction of each of the grid's cells falling within the
        jet's hard boundary define by w(r) (see RaJePy.maths.geometry.w_r
        method), or 'fill factors'
        """
        if self._ff is not None:
            return self._ff

        # TODO: Have disabled grid reflection due to correction of
        #  mgeom.xyz_to_rwp method to handle inclinations and position angles in
        #  the astronomically correct sense. Therefore, need to propagate these
        #  changes into the reflection logic at some point, for efficiency
        # # Establish reflective symmetries present, if any, to reduce
        # # computation time by reflection of coordinates/ffs/areas about
        # # relevant axes
        # refl_sym_x = False  # Reflective symmetry about x-axis?
        # refl_sym_y = False  # Reflective symmetry about y-axis?
        # refl_sym_z = False  # Reflective symmetry about z-axis?
        # refl_axes = []  # List holding reflected axes for use with np arrays
        #
        # if self.params["geometry"]["inc"] == 90.:
        #     if self.params["geometry"]["pa"] == 0.:
        #         refl_sym_x = refl_sym_y = refl_sym_z = True
        #         refl_axes = [0, 1, 2]
        #     else:
        #         refl_sym_y = True
        #         refl_axes = [1]
        # else:
        #     if self.params["geometry"]["pa"] == 0.:
        #         refl_sym_x = True
        #         refl_axes = [0]
        #
        # # Set up coordinate grids in x, y, z based upon axial reflective
        # # symmetries present given the provided inclination and position angle
        # if any((refl_sym_x, refl_sym_y, refl_sym_z)):
        #     if all((refl_sym_x, refl_sym_y, refl_sym_z)):
        #         xx, yy, zz = [_[int(self.nx / 2):,
        #                         int(self.ny / 2):,
        #                         int(self.nz / 2):] for _ in self.grid]
        #     else:
        #         if refl_sym_x:
        #             xx, yy, zz = [_[int(self.nx / 2):, :, :]
        #                           for _ in self.grid]
        #         elif refl_sym_y:
        #             xx, yy, zz = [_[:, int(self.ny / 2):, :]
        #                           for _ in self.grid]
        #         else:
        #             err_msg = u"Grid symmetry not understood for i = {:.0f}" \
        #                       u"\u00B0 and \u03B8={:.0f}\u00B0"
        #             err_msg = err_msg.format(self.params["geometry"]["inc"],
        #                                      self.params["geometry"]["pa"])
        #             raise ValueError(err_msg)
        #
        # else:
        #     xx, yy, zz = self.grid

        # xx, yy, zz = self.grid

        if self.log:
            self._log.add_entry(mtype="INFO",
                                entry="Calculating cells' fill "
                                      "factors/projected areas")

        else:
            print("INFO: Calculating cells' fill factors/projected areas")

        # Assign to local variables for readability
        w_0 = self.params['geometry']['w_0']
        r_0 = self.params['geometry']['r_0']
        mod_r_0 = self.params['geometry']['mod_r_0']
        eps = self.params['geometry']['epsilon']
        inc = self.params['geometry']['inc']
        pa = self.params['geometry']['pa']
        cs = self.csize

        ffs = np.zeros(np.shape(self.xx))
        areas = np.zeros(
            np.shape(self.xx))  # Areas as projected on to the y-axis
        # diag = np.sqrt(cs ** 2. * 3.)  # Diagonal dimensions of cells (au)
        # nvoxels = np.prod(np.shape(self.xx))
        # count = 0
        # progress = -1
        then = time.time()

        n_verts_inside = np.zeros(np.shape(self.xx), dtype=int)
        verts = ((0., 0., 0.), (cs, 0., 0.), (0., cs, 0.), (cs, cs, 0.),
                 (0., 0., cs), (cs, 0., cs), (0., cs, cs), (cs, cs, cs))

        for dx, dy, dz in verts:
            rv, wv = mgeom.xyz_to_rwp(self.xx + dx, self.yy + dy,
                                      self.zz + dz, inc, pa)[:2]
            wrv = mgeom.w_r(rv, w_0, mod_r_0, r_0, eps)
            n_verts_inside = np.where((wrv >= wv) & (np.abs(rv) >= r_0),
                                      n_verts_inside + 1, n_verts_inside)
        ffs = np.where(n_verts_inside == 8, 1.0, ffs)
        ffs = np.where((0 < n_verts_inside) & (n_verts_inside < 8), 0.5, ffs)
        areas = np.where(0 < n_verts_inside, 1.0, areas)

        # for idxy, yplane in enumerate(self.zz):
        #     for idxx, xrow in enumerate(yplane):
        #         for idxz, z in enumerate(xrow):
        #             count += 1
        #             # Does the cell definitely lie outside of the jet
        #             # boundary? Yes if w-coordinate is more than the cells'
        #             # full diagonal dimension away from the jet's width at
        #             # the cells' r-coordinate
        #             wr = mgeom.w_r(self.rr[idxy][idxx][idxz],
        #                            w_0, mod_r_0, r_0, eps)  # Removed
        #                                                     # np.abs(r) here
        #             if (self.ww[idxy][idxx][idxz] - 0.5 * diag) > wr:
        #                 continue
        #
        #             # Voxel blfc coordinate
        #             x, y = (self.xx[idxy][idxx][idxz],
        #                     self.yy[idxy][idxx][idxz])
        #
        #             # Voxel's vertices' coordinates
        #             verts = np.array([(x, y, z), (x + cs, y, z),
        #                               (x, y + cs, z), (x + cs, y + cs, z),
        #                               (x, y, z + cs), (x + cs, y, z + cs),
        #                               (x, y + cs, z + cs),
        #                               (x + cs, y + cs, z + cs)])
        #
        #             # Cell-vertices' r, w and phi coordinates
        #             rv, wv, pv = mgeom.xyz_to_rwp(verts[::, 0], verts[::, 1],
        #                                           verts[::, 2], inc, pa)
        #             wr = mgeom.w_r(rv, w_0, mod_r_0, r_0, eps)  # Removed
        #                                                         # np.abs(r)
        #                                                         # here
        #             verts_inside = (wv <= wr) & (np.abs(rv) >= r_0)
        #
        #             if np.sum(verts_inside) == 8:
        #                 ff = 1.
        #                 area = 1.
        #             elif np.sum(verts_inside) == 0:
        #                 continue
        #             else:
        #                 # TODO: Cells at base of jet need to accommodate for
        #                 #  r_0 properly. Value of 0.5 for ff and area will
        #                 #  not do
        #                 # Take average values for fill factor/projected areas
        #                 ff = .5
        #                 area = 1.0
        #
        #             ffs[idxy][idxx][idxz] = ff
        #             areas[idxy][idxx][idxz] = area
        #
        #         # Progress bar
        #         new_progress = int(count / nvoxels * 100)  #
        #         if new_progress > progress:
        #             progress = new_progress
        #             pblen = get_terminal_size().columns - 1
        #             pblen -= 16  # 16 non-varying characters
        #             s = '[' + ('=' * (int(progress / 100 * pblen) - 1)) + \
        #                 ('>' if int(progress / 100 * pblen) > 0 else '') + \
        #                 (' ' * int(pblen - int(progress / 100 * pblen))) + ']'
        #             # s += format(int(progress), '3') + '% complete'
        #             if progress != 0.:
        #                 t_sofar = (time.time() - then)
        #                 try:
        #                     rate = progress / t_sofar
        #                     secs_left = (100. - progress) / rate
        #                     s += time.strftime('%Hh%Mm%Ss left',
        #                                        time.gmtime(secs_left))
        #                 except ZeroDivisionError:
        #                     s += '  h  m  s left'
        #             else:
        #                 s += '  h  m  s left'
        #             print('\r' + s, end='' if progress < 100 else '\n')

        now = time.time()
        if self.log:
            self.log.add_entry(mtype="INFO",
                               entry=time.strftime('Finished in %Hh%Mm%Ss',
                                                   time.gmtime(now - then)))
        else:
            print(time.strftime('INFO: Finished in %Hh%Mm%Ss',
                                time.gmtime(now - then)))

        # TODO: Have disabled grid reflection due to correction of
        #  mgeom.xyz_to_rwp method to handle inclinations and position angles in
        #  the astronomically correct sense. Therefore, need to propagate these
        #  changes into the reflection logic at some point, for efficiency
        # # Reflect in x, y and z axes
        # for axis in refl_axes:
        #     ffs = np.append(np.flip(ffs, axis=axis), ffs, axis=axis)
        #     areas = np.append(np.flip(areas, axis=axis), areas, axis=axis)

        # Included as there are some, presumed floating point errors giving
        # fill factors of ~1e-15 on occasion
        ffs = np.where(ffs > 1e-6, ffs, np.NaN)
        areas = np.where(areas > 1e-6, areas, np.NaN)

        self._ff = ffs
        self._areas = areas

        return self._ff

    @property
    def areas(self) -> Union[None, np.ndarray]:
        """
        Areas of jet-filled portion of cells as projected on to the y-axis
        (hopefully, custom orientations will address this so area is as
        projected on to a surface whose normal points to the observer)
        """
        # if "_areas" in self.__dict__.keys() and self._areas is not None:
        if self._areas is not None:
            return self._areas

        _ = self.fill_factor  # Areas calculated as part of fill factors

        return self._areas

    @property
    def mass(self) -> np.ndarray:
        if self._m is not None:
            return self._m

        w_0 = self.params['geometry']['w_0'] / self.params['grid']['c_size']
        r_0 = self.params['geometry']['r_0'] / self.params['grid']['c_size']
        eps = self.params['geometry']['epsilon']

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
                # vol_zlayer = 0.
                # mass_slice = 0.
                continue

            ffs_zlayer = self.fill_factor[:, :, idz]
            m_cell = mass_slice / vol_zlayer  # kg / cell
            ms_zlayer = ffs_zlayer * m_cell

            ms[:, :, idz] = ms_zlayer

        ms = (self.number_density * (self.csize * con.au * 1e2) ** 3. *
              self.params['properties']['mu'] * mphys.atomic_mass('H') *
              self.fill_factor)

        ms = np.where(self.fill_factor > 0, ms, np.NaN)
        self._m = ms

        return self._m

    @mass.setter
    def mass(self, new_ms: np.ndarray):
        self._m = new_ms

    @property
    def ts(self) -> np.ndarray:
        """
        Time from launch of material in cell compared to current model time in
        seconds
        """
        if self._ts is not None:
            return self.time - self._ts

        r_0 = self.params['geometry']['r_0']
        r = np.abs(self.rr)
        r = np.where((r < r_0) & ((r + self.csize / 2.) >= r_0),
                     (r_0 + r + self.csize / 2.) / 2., r)

        ts = mgeom.t_rw(r, self.ww, self.params) * con.year
        self.ts = ts

        return self.ts

    @ts.setter
    def ts(self, new_ts: np.ndarray):
        self._ts = new_ts

    @property
    def chi_xyz(self) -> np.ndarray:
        """
        Chi factor (the burst factor) as a function of position.
        """
        return self.jml_t(self.ts) / self._ss_jml

    @property
    def number_density(self) -> np.ndarray:
        if self._nd is not None:
            return self._nd * self.chi_xyz

        r1 = self.params["target"]["R_1"]
        mr0 = self.params['geometry']['mod_r_0']
        r0 = self.params['geometry']['r_0']
        q_n = self.params["power_laws"]["q_n"]
        q_nd = self.params["power_laws"]["q^d_n"]
        n_0 = self.params["properties"]["n_0"]

        r = np.abs(self.rr)
        r = np.where((r < r0) & ((r + self.csize / 2.) >= r0),
                     (r0 + r + self.csize / 2.) / 2., r)
        # nd = n_0 * mgeom.rho(self.rr, r0, mr0) ** q_n * \
        #      (self.rreff / r1) ** q_nd
        nd = mgeom.cell_value(n_0, mgeom.rho(r, r0, mr0), self.rreff,
                              r1, q_n, q_nd)
        nd = np.where(self.fill_factor > 0, nd, np.NaN)
        nd = np.where(nd == 0, np.NaN, nd)

        self.number_density = np.nan_to_num(nd, nan=np.NaN, posinf=np.NaN,
                                            neginf=np.NaN)

        return self.number_density

    @number_density.setter
    def number_density(self, new_nds: np.ndarray):
        self._nd = new_nds

    @property
    def mass_density(self) -> np.ndarray:
        """
        Mass density in g cm^-3
        """
        av_m_particle = self.params['properties']['mu'] * mphys.atomic_mass("H")

        return av_m_particle * 1e3 * self.number_density  # g cm^-3

    @property
    def ion_fraction(self) -> np.ndarray:
        if self._xi is not None:
            return self._xi

        r_1 = self.params["target"]["R_1"]
        mod_r_0 = self.params['geometry']['mod_r_0']
        r_0 = self.params['geometry']['r_0']
        q_x = self.params["power_laws"]["q_x"]
        q_xd = self.params["power_laws"]["q^d_x"]
        x_0 = self.params["properties"]["x_0"]

        r = np.abs(self.rr)
        r = np.where((r < r_0) & ((r + self.csize / 2.) >= r_0),
                     (r_0 + r + self.csize / 2.) / 2., r)

        # xi = x_0 * mgeom.rho(r, r_0, mod_r_0) ** q_x * \
        #      (self.rreff / r_1) ** q_xd
        xi = mgeom.cell_value(x_0, mgeom.rho(r, r_0, mod_r_0), self.rreff,
                              r_1, q_x, q_xd)
        xi = np.where(self.fill_factor > 0, xi, np.NaN)
        xi = np.where(xi == 0, np.NaN, xi)

        self.ion_fraction = np.nan_to_num(xi, nan=np.NaN, posinf=np.NaN,
                                          neginf=np.NaN)

        return self.ion_fraction

    @ion_fraction.setter
    def ion_fraction(self, new_xis: np.ndarray):
        self._xi = new_xis

    @property
    def temperature(self) -> np.ndarray:
        """
        Temperature (in Kelvin)
        """
        if self._temp is not None:
            return self._temp

        r_1 = self.params["target"]["R_1"]
        mod_r_0 = self.params['geometry']['mod_r_0']
        r_0 = self.params['geometry']['r_0']
        q_t = self.params["power_laws"]["q_T"]
        q_td = self.params["power_laws"]["q^d_T"]
        temp_0 = self.params["properties"]["T_0"]

        r = np.abs(self.rr) * con.au * 1e2
        r = np.where((r < r_0) & ((r + self.csize / 2.) >= r_0),
                     (r_0 + r + self.csize / 2.) / 2., r)

        temp = mgeom.cell_value(temp_0, mgeom.rho(r, r_0, mod_r_0), self.rreff,
                                r_1, q_t, q_td)
        temp = np.where(self.fill_factor > 0, temp, np.NaN)
        temp = np.where(temp == 0, np.NaN, temp)

        self.temperature = np.nan_to_num(temp, nan=np.NaN, posinf=np.NaN,
                                         neginf=np.NaN)

        return self.temperature

        # z = np.abs(self.rr)
        # a = z - 0.5 * self.csize
        # b = z + 0.5 * self.csize
        #
        # a = np.where(b <= self.params['geometry']['r_0'], np.NaN, a)
        # b = np.where(b <= self.params['geometry']['r_0'], np.NaN, b)
        #
        # a = np.where(a <= self.params['geometry']['r_0'],
        #              self.params['geometry']['r_0'], a)
        #
        # def indefinite_integral(z):
        #     num_p1 = self.params['properties']['T_0'] * \
        #              self.params["geometry"]["mod_r_0"]
        #     num_p2 = ((z + self.params["geometry"]["mod_r_0"] -
        #                self.params["geometry"]["r_0"]) /
        #               (self.params["geometry"]["mod_r_0"]))
        #     num_p2 = num_p2 ** (self.params["power_laws"]["q_T"] + 1.)
        #     den = self.params["power_laws"]["q_T"] + 1.
        #     return num_p1 * num_p2 / den
        #
        # ts = indefinite_integral(b) - indefinite_integral(a)
        # ts /= b - a
        # ts = np.where(self.fill_factor > 0., ts, np.NaN)
        # self.temperature = ts
        #
        # return self.temperature

    @temperature.setter
    def temperature(self, new_ts: np.ndarray):
        self._temp = new_ts

    @property
    def pressure(self) -> np.ndarray:
        """
        Pressure in Barye (or dyn cm^-2)
        """
        return self.number_density * self.temperature * con.k * 1e7

    @property
    def vel(self) -> np.ndarray:
        """
        Velocity components in km/s
        """
        if self._v is not None:
            return self._v

        # r = np.abs(self.rr)
        #
        # r_0 = self.params['geometry']['r_0']
        # mr0 = self.params['geometry']['mod_r_0']
        #
        # a = r - 0.5 * self.csize
        # b = r + 0.5 * self.csize
        #
        # a = np.where(b <= r_0, np.NaN, a)
        # b = np.where(b <= r_0, np.NaN, b)
        #
        # a = np.where(a <= r_0, r_0, a)
        #
        # def indefinite_integral(_r):
        #     num_p1 = self.params['properties']['v_0'] * mr0
        #     num_p2 = (_r + mr0 - r_0) / mr0
        #     num_p2 = num_p2 ** (self.params["power_laws"]["q_v"] + 1.)
        #     den = self.params["power_laws"]["q_v"] + 1.
        #     return num_p1 * num_p2 / den
        #
        # vz = indefinite_integral(b) - indefinite_integral(a)
        # vz /= b - a
        #
        # vz = np.where(self.fill_factor > 0., vz, np.NaN) * np.sign(self.rr)

        r_1 = self.params["target"]["R_1"]
        mod_r_0 = self.params['geometry']['mod_r_0']
        r_0 = self.params['geometry']['r_0']
        mr0 = self.params['geometry']['mod_r_0']
        q_v = self.params["power_laws"]["q_v"]
        q_vd = self.params["power_laws"]["q^d_v"]
        v_0 = self.params["properties"]["v_0"]

        r = np.abs(self.rr)
        r = np.where((r < r_0) & ((r + self.csize / 2.) >= r_0),
                     (r_0 + r + self.csize / 2.) / 2., r)

        # vz = v_0 * mgeom.rho(r, r_0, mod_r_0) ** q_v * \
        #      (self.rreff / r_1) ** q_vd
        vz = mgeom.cell_value(v_0, mgeom.rho(r, r_0, mod_r_0), self.rreff, r_1,
                              q_v, q_vd)
        vz = np.where(self.fill_factor > 0, vz, np.NaN)
        vz = np.where(vz == 0, np.NaN, vz)

        vz = np.nan_to_num(vz, nan=np.NaN, posinf=np.NaN,
                           neginf=np.NaN) * np.sign(self.rr)

        vr = mphys.v_rot(self.rr, self.rreff, mgeom.rho(self.rr, r_0, mr0),
                         self.params['geometry']['epsilon'],
                         self.params['target']['M_star'])

        # x/y components of rotational velocity
        rotation_direction = self.params["geometry"]["rotation"].lower()
        vx = -vr * np.sin(self.pp) * (1 if rotation_direction == 'ccw' else -1)
        vy = vr * np.cos(self.pp) * (1 if rotation_direction == 'ccw' else -1)

        vx = np.where(self.fill_factor > 0., vx, np.NaN)
        vy = np.where(self.fill_factor > 0., vy, np.NaN)
        vz = np.where(self.fill_factor > 0., vz, np.NaN)

        i = self.params["geometry"]["inc"]
        pa = self.params["geometry"]["pa"]

        # vxs = np.empty(np.shape(self.xx))
        # vys = np.empty(np.shape(self.xx))
        # vzs = np.empty(np.shape(self.xx))
        # vs = np.stack([vx, vy, vz], axis=3)
        # for idxx, plane in enumerate(vs):
        #     for idxy, column in enumerate(plane):
        #         for idxz, v in enumerate(column):
        #             #x, y, z = rot_x.dot(rot_y.dot(v))
        #             x, y, z = mgeom.xyz_rotate(*v, 90. - i, -pa, order='xy')
        #             vxs[idxx][idxy][idxz] = x
        #             vys[idxx][idxy][idxz] = y
        #             vzs[idxx][idxy][idxz] = z
        vxs, vys, vzs = mgeom.xyz_rotate(vx, vy, vz, 90. - i, -pa, order='xy')
        self.vel = (vxs, vys + self.params["target"]["v_lsr"], vzs)

        return self.vel

    @vel.setter
    def vel(self, new_vs: np.ndarray):
        self._v = new_vs

    def emission_measure(self,
                         savefits: Union[bool, str] = False) -> np.ndarray:
        """
        Emission measure as viewed along the y-axis (pc cm^-6)

        Parameters
        ----------
        savefits : bool, str
            False or full path to save calculated optical depths as .fits file

        Returns
        -------
        ems : numpy.ndarray
            Emission measures as viewed along y-axis
        """
        ems = (self.number_density * self.ion_fraction) ** 2. * \
              (self.csize * con.au / con.parsec *
               (self.fill_factor / self.areas))

        ems = np.nansum(ems, axis=self.los_axis)

        if savefits:
            self.save_fits(
                miscf.reorder_axes(ems, ra_axis=0, dec_axis=1),
                savefits, 'em'
            )

        return ems

    def optical_depth_rrl(self, rrl: str,
                          freq: Union[float, npt.ArrayLike],
                          lte: bool = True,
                          savefits: Union[bool, str] = False,
                          collapse: bool = True) -> np.ndarray:
        """
        Return RRL optical depth as viewed along the y-axis

        Parameters
        ----------
        rrl : str
            Notation for RRL e.g. H58a, He42b etc.
        freq : float
            Frequency of observation (Hz).
        lte : bool
            Whether to assume local thermodynamic equilibrium or not. Default is
            True
        savefits : bool, str
            False or full path to save calculated optical depths as .fits file
        collapse : bool
            Whether to sum the optical depths along the line of sight axis,
            or return the 3-dimensional array of optical depts (default is True)
        Returns
        -------
        tau_rrl : numpy.ndarray
            RRL optical depths as viewed along y-axis (if collapse is True),
            or the 3-dimensional array of optical depths (if collapse is False)
        """
        # #################### RRL Information ############################### #
        element, rrl_n, rrl_dn = mrrl.rrl_parser(rrl)
        rest_freq = mphys.doppler_shift(mrrl.rrl_nu_0(element, rrl_n, rrl_dn),
                                        self.vel[1])

        n_es = self.number_density * self.ion_fraction

        rrl_fwhm_thermal = mrrl.deltanu_g(rest_freq, self.temperature, element)
        fn1n2 = mrrl.f_n1n2(rrl_n, rrl_dn)
        en = mrrl.energy_n(rrl_n, element)
        z_atom = mphys.z_number(element)
        rrl_fwhm_stark = mrrl.deltanu_l(n_es, rrl_n, rrl_dn)

        phi_v = mrrl.phi_voigt_nu(rest_freq, rrl_fwhm_stark, rrl_fwhm_thermal)

        if not np.isscalar(freq):
            if collapse:
                tau_rrl = np.empty((np.shape(freq)[0], self.nx, self.nz))
            else:
                tau_rrl = np.empty((np.shape(freq)[0], *np.shape(self.xx)))

            for idx, f in enumerate(freq):
                kappa_rrl_lte = mrrl.kappa_l(f, rrl_n, fn1n2, phi_v(f),
                                             n_es,
                                             mrrl.ni_from_ne(n_es, element),
                                             self.temperature, z_atom, en)
                taus = kappa_rrl_lte * (self.csize * con.au * 1e2 *
                                        (self.fill_factor / self.areas))
                if collapse:
                    taus = np.nansum(taus, axis=self.los_axis)

                tau_rrl[idx] = taus

            if savefits:
                if collapse:
                    self.save_fits(
                        miscf.reorder_axes(tau_rrl, ra_axis=1, dec_axis=2,
                                           axis3=0, axis3_type='freq'),
                        savefits, 'tau', freq
                    )
                else:
                    self.save_fits(
                        miscf.reorder_axes(tau_rrl, ra_axis=1, dec_axis=3,
                                           axis3=2, axis3_type='y',
                                           axis4=0, axis4_type='freq'),
                        savefits, 'tau', freq
                    )

        else:
            kappa_rrl_lte = mrrl.kappa_l(freq, rrl_n, fn1n2, phi_v(freq),
                                         n_es, mrrl.ni_from_ne(n_es, element),
                                         self.temperature, z_atom, en)
            tau_rrl = kappa_rrl_lte * (self.csize * con.au * 1e2 *
                                       (self.fill_factor / self.areas))

            if collapse:
                tau_rrl = np.nansum(tau_rrl, axis=self.los_axis)

            if savefits:
                if collapse:
                    self.save_fits(
                        miscf.reorder_axes(tau_rrl, ra_axis=0, dec_axis=1),
                        savefits, 'tau', freq
                    )
                else:
                    self.save_fits(
                        miscf.reorder_axes(tau_rrl, ra_axis=0, dec_axis=2,
                                           axis3=1, axis3_type='y'),
                        savefits, 'tau', freq
                    )

        return tau_rrl

    def intensity_rrl(self, rrl: str,
                      freq: Union[float, Union[np.ndarray, List[float]]],
                      lte: bool = True,
                      savefits: Union[bool, str] = False) -> np.ndarray:
        """
        Radio intensity as viewed along x-axis (in W m^-2 Hz^-1 sr^-1)

        Parameters
        ----------
        rrl : str
            Notation for RRL e.g. H58a, He42b etc.
        freq : float
            Frequency of observation (Hz).
        lte : bool
            Whether to assume local thermodynamic equilibrium or not. Default is
            True
        savefits : bool, str
            False or full path to save calculated optical depths as .fits file
        Returns
        -------
        i_rrl : numpy.ndarray
            RRL intensities as viewed along y-axis
        """
        av_temp = np.nanmean(np.where(self.temperature > 0.,
                                      self.temperature, np.NaN),
                             axis=self.los_axis)

        if lte:
            line_intensity = mrrl.line_intensity_lte
        else:
            raise ValueError("Non-LTE RRL calculations not yet supported")
            # line_intensity = mrrl.line_intensity_nonlte

        if not np.isscalar(freq):  # isinstance(freq, Iterable):
            ints_rrl = np.empty((np.shape(freq)[0], self.nx, self.nz))
            for idx, nu in enumerate(freq):
                tau_rrl = self.optical_depth_rrl(rrl, freq, lte=lte,
                                                 collapse=True)
                tau_ff = self.optical_depth_ff(freq, collapse=True)
                i_rrl = line_intensity(freq, av_temp, tau_ff, tau_rrl)
                ints_rrl[idx] = i_rrl
            if savefits:
                self.save_fits(
                    miscf.reorder_axes(ints_rrl, ra_axis=1, dec_axis=2,
                                       axis3=0, axis3_type='freq'),
                    savefits, 'intensity', freq
                )

        else:
            tau_rrl = self.optical_depth_rrl(rrl, freq, lte=lte, collapse=True)
            tau_ff = self.optical_depth_ff(freq, collapse=True)
            ints_rrl = line_intensity(freq, av_temp, tau_ff, tau_rrl)

            if savefits:
                self.save_fits(
                    miscf.reorder_axes(ints_rrl, ra_axis=0, dec_axis=1),
                    savefits, 'intensity', freq
                )

        return ints_rrl

    def flux_rrl(self, rrl: str,
                 freq: Union[float, Union[np.ndarray, List[float]]],
                 lte: bool = True, contsub: bool = True,
                 savefits: Union[bool, str] = False) -> np.ndarray:
        """
        Return RRL flux (in Jy). Note these are the continuum-subtracted fluxes

        Parameters
        ----------
        rrl : str
            Notation for RRL e.g. H58a, He42b etc.
        freq : float, np.ndarray, list
            Frequency of observation (Hz).
        lte : bool
            Whether to assume local thermodynamic equilibrium or not. Default is
            True
        contsub : bool
            Whether to return the continuum subtracted fluxes or not (default is
            True)
        savefits : bool, str
            False or full path to save calculated optical depths as .fits file

        Returns
        -------
        flux_rrl : numpy.ndarray
            RRL fluxes as viewed along y-axis.
        """
        if not np.isscalar(freq):  # isinstance(freq, Iterable):
            fluxes = np.empty((np.shape(freq)[0], self.nx, self.nz))
            for idx, nu in enumerate(freq):
                i_rrl = self.intensity_rrl(rrl, nu, lte=lte, savefits=False)
                flux = i_rrl * np.arctan((self.csize * con.au) /
                                         (self.params["target"]["dist"] *
                                          con.parsec)) ** 2. / 1e-26
                if not contsub:
                    flux += self.flux_ff(nu)
                fluxes[idx] = flux

            if savefits:
                self.save_fits(
                    miscf.reorder_axes(fluxes, ra_axis=1, dec_axis=2, axis3=0,
                                       axis3_type='freq'),
                    savefits, 'flux', freq
                )

        else:
            i_rrl = self.intensity_rrl(rrl, freq, savefits=False)
            fluxes = i_rrl * np.arctan((self.csize * con.au) /
                                       (self.params["target"]["dist"] *
                                        con.parsec)) ** 2. / 1e-26
            if not contsub:
                fluxes += self.flux_ff(freq)

            if savefits:
                self.save_fits(
                    miscf.reorder_axes(fluxes, ra_axis=0, dec_axis=1),
                    savefits, 'flux', freq
                )

        return fluxes

    def optical_depth_ff(self,
                         freq: Union[float, Union[np.ndarray, List[float]]],
                         savefits: Union[bool, str] = False,
                         collapse: bool = True) -> np.ndarray:
        """
        Return free-free optical depth as viewed along the y-axis

        Parameters
        ----------
        freq : float, np.ndarray, list
            Frequency of observation (Hz).
        savefits : bool, str
            False or full path to save calculated optical depths as .fits file
        collapse : bool
            Whether to sum the optical depths along the line of sight axis,
            or return the 3-dimensional array of optical depts (default is True)
        Returns
        -------
        tau_ff : numpy.ndarray
            Optical depths as viewed along y-axis.

        """
        n_es = self.number_density * self.ion_fraction

        # Equation 1.26 and 5.19b of Rybicki and Lightman (cgs). Averaged
        # path length through voxel is volume / projected area
        if not np.isscalar(freq):  # isinstance(freq, Iterable):
            if collapse:
                tff = np.empty((np.shape(freq)[0], self.nx, self.nz))
            else:
                tff = np.empty((np.shape(freq)[0], *np.shape(self.xx)))
            for idx, nu in enumerate(freq):
                # Gaunt factors of van Hoof et al. (2014). Use if constant
                # temperature as computation via this method across a grid
                # takes too long Free-free Gaunt factors
                if self.params['power_laws']['q_T'] == 0.:
                    gff = mphys.gff(nu, self.params['properties']['T_0'])

                # Equation 1 of Reynolds (1986) otherwise as an approximation
                else:
                    gff = 11.95 * self.temperature ** 0.15 * nu ** -0.1

                tau = (0.018 * self.temperature ** -1.5 * nu ** -2. *
                       n_es ** 2. * (self.csize * con.au * 1e2 *
                                     (self.fill_factor / self.areas)) * gff)
                if collapse:
                    tau = np.nansum(tau, axis=self.los_axis)
                tff[idx] = tau

                if savefits:
                    if collapse:
                        self.save_fits(
                            miscf.reorder_axes(tff, ra_axis=1, dec_axis=2,
                                               axis3=0, axis3_type='freq'),
                            savefits, 'tau', freq
                        )
                    else:
                        self.save_fits(
                            miscf.reorder_axes(tff, ra_axis=1, dec_axis=3,
                                               axis3=2, axis3_type='y',
                                               axis4=0, axis4_type='freq'),
                            savefits, 'tau', freq
                        )

        else:
            # Gaunt factors of van Hoof et al. (2014). Use if constant temp
            # as computation via this method across a grid takes too long
            # Free-free Gaunt factors
            if self.params['power_laws']['q_T'] == 0.:
                gff = mphys.gff(freq, self.params['properties']['T_0'])

            # Equation 1 of Reynolds (1986) otherwise as an approximation
            else:
                gff = 11.95 * self.temperature ** 0.15 * freq ** -0.1
            tff = (0.018 * self.temperature ** -1.5 * freq ** -2. *
                   n_es ** 2. * (self.csize * con.au * 1e2 *
                                 (self.fill_factor / self.areas)) * gff)

            if collapse:
                tff = np.nansum(tff, axis=self.los_axis)

            if savefits:
                if collapse:
                    self.save_fits(
                        miscf.reorder_axes(tff, ra_axis=0, dec_axis=1),
                        savefits, 'tau', freq
                    )
                else:
                    self.save_fits(
                        miscf.reorder_axes(tff, ra_axis=0, dec_axis=2,
                                           axis3=1, axis3_type='y'),
                        savefits, 'tau', freq
                    )

        return tff

    def intensity_ff(self, freq: Union[float, Union[np.ndarray, List[float]]],
                     savefits: Union[bool, str] = False) -> np.ndarray:
        """
        Radio intensity as viewed along y-axis (in W m^-2 Hz^-1 sr^-1)

        Parameters
        ----------
        freq : float, np.ndarray, list
            Frequency of observation (Hz).
        savefits : bool, str
            False or full path to save calculated optical depths as .fits file

        Returns
        -------
        ints_ff : numpy.ndarray
            Optical depths as viewed along y-axis.
        """
        ts = self.temperature

        if not np.isscalar(freq):  # isinstance(freq, Iterable):
            ints_ff = np.empty((np.shape(freq)[0], self.nx, self.nz))
            for idx, nu in enumerate(freq):
                temp_b = np.nanmean(np.where(ts > 0., ts, np.NaN),
                                    axis=self.los_axis) * \
                         (1. - np.exp(-self.optical_depth_ff(nu)))

                iff = 2. * nu ** 2. * con.k * temp_b / con.c ** 2.
                ints_ff[idx] = iff
            if savefits:
                self.save_fits(
                    miscf.reorder_axes(ints_ff, ra_axis=1, dec_axis=2,
                                       axis3=0, axis3_type='freq'),
                    savefits, 'intensity', freq
                )
        else:
            temp_b = np.nanmean(np.where(ts > 0., ts, np.NaN),
                                axis=self.los_axis) * \
                     (1. - np.exp(-self.optical_depth_ff(freq)))

            ints_ff = 2. * freq ** 2. * con.k * temp_b / con.c ** 2.

            if savefits:
                self.save_fits(
                    miscf.reorder_axes(ints_ff, ra_axis=0, dec_axis=1),
                    savefits, 'intensity', freq
                )

        return ints_ff

    def flux_ff(self, freq: Union[float, Union[np.ndarray, List[float]]],
                savefits: Union[bool, str] = False) -> np.ndarray:
        """
        Return flux (in Jy)

        Parameters
        ----------
        freq : float, np.ndarray, list
            Frequency of observation (Hz).
        savefits : bool, str
            False or full path to save calculated optical depths as .fits file

        Returns
        -------
        flux_ff : numpy.ndarray
            Fluxes as viewed along y-axis.
        """
        if not np.isscalar(freq):  # isinstance(freq, Iterable):
            fluxes = np.empty((np.shape(freq)[0], self.nx, self.nz))
            for idx, nu in enumerate(freq):
                ints = self.intensity_ff(nu)
                fs = ints * np.arctan((self.csize * con.au) /
                                      (self.params["target"]["dist"] *
                                       con.parsec)) ** 2. / 1e-26
                fluxes[idx] = fs
            if savefits:
                self.save_fits(
                    miscf.reorder_axes(fluxes, ra_axis=1, dec_axis=2,
                                       axis3=0, axis3_type='freq'),
                    savefits, 'flux', freq
                )
        else:
            ints = self.intensity_ff(freq)
            fluxes = ints * np.arctan((self.csize * con.au) /
                                      (self.params["target"]["dist"] *
                                       con.parsec)) ** 2. / 1e-26

            if savefits:
                self.save_fits(
                    miscf.reorder_axes(fluxes, ra_axis=0, dec_axis=1),
                    savefits, 'flux', freq
                )

        return fluxes

    def save_fits(self, data: np.ndarray, filename: str, image_type: str,
                  freq: Union[float, list, np.ndarray, None] = None):
        """
        Save .fits file of input data. For the data array, axis-0 must
        correspond to declination, axis-1 to right ascension and axis-3 to
        e.g. frequency. It is the responsibility of the caller to ensure axes
        are in the correct order (use of numpy.swapaxes will be helpful here)

        Parameters
        ----------
        data : numpy.array
            2-D/3-D numpy array of image data.
        filename: str
            Full path to save .fits image to
        image_type : str
            One of 'flux', 'tau' or 'em'. The type of image data saved.
        freq : float
            Radio frequency of image (ignored if image_type is 'em')

        Returns
        -------
        None.

        Raises
        ------
        ValueError
            If image_type is not 'flux', 'tau', or 'em'
        """
        if image_type not in ('flux', 'tau', 'em', 'intensity'):
            raise ValueError("arg image_type must be one of 'flux', 'tau' or "
                             "'em'")

        c = SkyCoord(self.params['target']['ra'],
                     self.params['target']['dec'],
                     unit=(u.hourangle, u.degree), frame='fk5')

        csize_deg = np.degrees(np.arctan(self.csize * con.au /
                                         (self.params['target']['dist'] *
                                          con.parsec)))

        ndims = len(np.shape(data))

        if ndims not in (2, 3):
            raise ValueError(f"Unexpected number of data dimensions ({ndims})")

        hdu = fits.PrimaryHDU(data)

        # hdu = fits.PrimaryHDU(np.array([data]))
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

        if image_type in ('flux', 'tau', 'intensity'):
            if ndims == 3:
                nchan = len(freq)
                if nchan != 1:
                    chan_width = freq[1] - freq[0]
                else:
                    chan_width = 1.
                hdr['CTYPE3'] = 'FREQ'
                hdr.comments['CTYPE3'] = 'Spectral axis (frequency)'
                hdr['CRPIX3'] = nchan / 2. + 0.5
                hdr.comments['CRPIX3'] = 'Reference frequency (channel number)'
                hdr['CRVAL3'] = freq[len(freq) // 2 - 1] + chan_width / 2
                hdr.comments['CRVAL3'] = 'Reference frequency (Hz)'
                hdr['CDELT3'] = chan_width
                hdr.comments['CDELT3'] = 'Frequency increment (Hz)'
            else:
                hdr['CDELT3'] = 1.
                hdr.comments['CDELT3'] = 'Frequency increment (Hz)'
                hdr['CRPIX3'] = 0.5
                hdr.comments['CRPIX3'] = 'Reference frequency (channel number)'
                hdr['CRVAL3'] = freq[0]
                hdr.comments['CRVAL3'] = 'Reference frequency (Hz)'

        if image_type == 'flux':
            hdr['BUNIT'] = 'Jy pixel^-1'
        elif image_type == 'intensity':
            hdr['BUNIT'] = 'W m^-2 Hz^-1 sr^-1'
        elif image_type == 'em':
            hdr['BUNIT'] = 'pc cm^-6'
        elif image_type == 'tau':
            hdr['BUNIT'] = 'dimensionless'

        s_hist = self.__str__().split('\n')
        hdr['HISTORY'] = (' ' * (72 - len(s_hist[0]))).join(s_hist)

        hdul.writeto(filename, overwrite=True)

        return None

    @property
    def log(self) -> Union[None, logger.Log]:
        return self._log

    @log.setter
    def log(self, new_log: Union[None, logger.Log]):
        self._log = new_log

    @property
    def csize(self) -> float:
        return self._csize

    @property
    def nx(self) -> int:
        return self._nx

    @property
    def ny(self) -> int:
        return self._ny

    @property
    def nz(self) -> int:
        return self._nz

    @property
    def params(self) -> Dict:
        return self._params

    @property
    def name(self) -> str:
        return self._name

    @property
    def ejections(self) -> Dict:
        return self._ejections

    @property
    def ss_jml(self):
        return self._ss_jml

    def save(self, filename):
        ps = {'params': self._params,
              'areas': None if self._areas is None else self.areas,
              'ffs': None if self._ff is None else self.fill_factor,
              'time': self.time,
              'log': self.log}
        self.log.add_entry("INFO", "Saving physical model to "
                                   "{}".format(filename))
        pickle.dump(ps, open(filename, "wb"))
        return None


class ContinuumRun:
    def __init__(self, dcy: str, year: float,
                 freq: Union[float, None] = None,
                 bandwidth: Union[float, None] = None,
                 chanwidth: Union[float, None] = None,
                 t_obs: Union[float, None] = None,
                 t_int: Union[float, None] = None,
                 tscop: Union[Tuple[str, str], None] = None):

        # Protected, 'immutable' attributes
        self._year = year
        self._dcy = dcy
        self._obs_type = 'continuum'
        self._freq = freq
        self._t_obs = t_obs
        self._t_int = t_int
        self._tscop = tscop
        self._products = {}
        self._results = {}

        # Default bandwidth/channel-width values to 1 Hz if not given
        if bandwidth is not None:
            self._bandwidth = bandwidth
        else:
            self._bandwidth = 1.

        if chanwidth is not None:
            self._chanwidth = chanwidth
        else:
            self._chanwidth = 1.

        # Public, 'mutable' attributes
        self.completed = False
        self.radiative_transfer = True
        self.simobserve = True

        if freq is None:
            self.radiative_transfer = False

        for val in (tscop, bandwidth, chanwidth, t_obs, t_int):
            if val is None:
                self.simobserve = False

    def __str__(self):
        hdr = ['Year', 'Type', 'Telescope', 't_obs', 't_int', 'Line',
               'Frequency', 'Bandwidth', 'Channel width',
               'Radiative Transfer?', 'Synthetic Obs.?', 'Completed?']
        units = ['yr', '', '', 's', 's', '', 'Hz', 'Hz', 'Hz', '', '', '']
        fmt = ['.2f', '', '', '.0f', '.0f', '', '.3e', '.3e', '.3e', '', '', '']
        val = [self._year, self._obs_type.capitalize(), self._tscop,
               self._t_obs, self._t_int, None,
               self._freq, self._bandwidth, self._chanwidth,
               self.radiative_transfer, self.simobserve, self.completed]

        for i, v in enumerate(val):
            if v is None:
                val[i] = '-'

        tab_head = []
        for i, h in enumerate(hdr):
            if units[i] != '':
                tab_head.append(h + '\n[' + units[i] + ']')
            else:
                tab_head.append(h)

        tab = tabulate.tabulate([val], tab_head, tablefmt="grid",
                                floatfmt=fmt)

        return tab

    @property
    def results(self) -> dict:
        """Quantitative results gleaned from products"""
        return self._results

    @results.setter
    def results(self, new_results: dict):
        if not isinstance(new_results, dict):
            raise TypeError("setter method for results attribute requires dict")
        self._results = new_results

    @results.deleter
    def results(self):
        del self._results

    @property
    def products(self) -> dict:
        """Any data products resulting from the executed run"""
        return self._products

    @products.setter
    def products(self, new_products: dict):
        if not isinstance(new_products, dict):
            raise TypeError("setter method for products attribute requires "
                            "dict")
        self._products = new_products

    @products.deleter
    def products(self):
        del self._products

    @property
    def obs_type(self) -> str:
        return self._obs_type

    @property
    def dcy(self) -> str:
        """Parent directory of the run"""
        return self._dcy

    @dcy.setter
    def dcy(self, path):
        self._dcy = path

    @property
    def model_dcy(self) -> str:
        """Directory containing model files"""
        return os.sep.join([self.dcy, f'Day{self.day}'])

    @property
    def rt_dcy(self) -> Union[str, None]:
        """Directory to contain radiative transfer data products/files"""
        if not self.radiative_transfer:
            return None
        else:
            return os.sep.join([self.model_dcy, miscf.freq_str(self.freq)])

    @property
    def year(self) -> float:
        return self._year

    @property
    def day(self) -> float:
        return int(self.year * 365.)

    @property
    def freq(self) -> float:
        return self._freq

    @property
    def bandwidth(self) -> float:
        return self._bandwidth

    @property
    def chanwidth(self) -> float:
        return self._chanwidth

    @property
    def t_obs(self) -> float:
        return self._t_obs

    @property
    def t_int(self) -> float:
        return self._t_int

    @property
    def tscop(self) -> Tuple[str, str]:
        return self._tscop

    @property
    def fits_flux(self) -> str:
        return self.rt_dcy + os.sep + '_'.join(['Flux', 'Day' + str(self.day),
                                                miscf.freq_str(self.freq)]) + \
               '.fits'

    @property
    def fits_tau(self) -> str:
        return self.rt_dcy + os.sep + '_'.join(['Tau', 'Day' + str(self.day),
                                                miscf.freq_str(self.freq)]) + \
               '.fits'

    @property
    def fits_em(self) -> str:
        return self.rt_dcy + os.sep + '_'.join(['EM', 'Day' + str(self.day),
                                                miscf.freq_str(self.freq)]) + \
               '.fits'

    @property
    def nchan(self) -> int:
        return int(self.bandwidth / self.chanwidth)

    @property
    def chan_freqs(self) -> np.ndarray:
        chan1 = self.freq - self.bandwidth / 2. + self.chanwidth / 2.
        return chan1 + np.arange(self.nchan) * self.chanwidth


class RRLRun(ContinuumRun):
    def __init__(self, dcy: str, year: float,
                 line: Union[str, None] = None,
                 bandwidth: Union[float, None] = None,
                 chanwidth: Union[float, None] = None,
                 t_obs: Union[float, None] = None,
                 t_int: Union[float, None] = None,
                 tscp: Union[Tuple[str, str], None] = None):
        self.line = line
        freq = mrrl.rrl_nu_0(*mrrl.rrl_parser(line))

        super().__init__(dcy, year, freq, bandwidth, chanwidth, t_obs, t_int,
                         tscp)

        self._obs_type = 'rrl'

    def __str__(self):
        hdr = ['Year', 'Type', 'Telescope', 't_obs', 't_int', 'Line',
               'Frequency', 'Bandwidth', 'Channel width',
               'Radiative Transfer?', 'Synthetic Obs.?', 'Completed?']
        units = ['yr', '', '', 's', 's', '', 'Hz', 'Hz', 'Hz', '', '', '']
        fmt = ['.2f', '', '', '.0f', '.0f', '', '.3e', '.3e', '.3e', '', '', '']
        val = [self._year, self._obs_type.capitalize(), self._tscop,
               self._t_obs, self._t_int, self.line,
               self._freq, self._bandwidth, self._chanwidth,
               self.radiative_transfer, self.simobserve, self.completed]

        for i, v in enumerate(val):
            if v is None:
                val[i] = '-'

        tab_head = []
        for i, h in enumerate(hdr):
            if units[i] != '':
                tab_head.append(h + '\n[' + units[i] + ']')
            else:
                tab_head.append(h)

        tab = tabulate.tabulate([val], tab_head, tablefmt="grid",
                                floatfmt=fmt)

        return tab

    @property
    def rt_dcy(self) -> Union[str, None]:
        """Directory to contain radiative transfer data products/files"""
        if not self.radiative_transfer:
            return None
        else:
            return os.sep.join([self.model_dcy, self.line])

    @property
    def fits_flux(self) -> str:
        return self.rt_dcy + os.sep + '_'.join(['Flux', 'Day' + str(self.day),
                                                self.line]) + '.fits'

    @property
    def fits_tau(self) -> str:
        return self.rt_dcy + os.sep + '_'.join(['Tau', 'Day' + str(self.day),
                                                self.line]) + '.fits'

    @property
    def fits_em(self) -> str:
        return self.rt_dcy + os.sep + '_'.join(['EM', 'Day' + str(self.day),
                                                self.line]) + '.fits'


class Pipeline:
    """
    Class to handle a creation of physical jet model, creation of .fits files
    and subsequent synthetic imaging via CASA.
    """

    @classmethod
    def load_pipeline(cls, load_file) -> 'Pipeline':
        """
        Loads pipeline from a previously saved state

        Parameters
        ----------
        cls : Pipeline
            DESCRIPTION.
        load_file : str
            Full path to saved ModelRun file (pickle file).

        Returns
        -------
        Instance of ModelRun initiated from save state.
        """
        home = os.path.expanduser('~')
        load_file = os.path.expanduser(load_file)
        loaded = pickle.load(open(load_file, 'rb'))

        for idx, run in enumerate(loaded['runs']):
            run.dcy = run.dcy.replace('~', home)
            loaded['runs'][idx] = run

        loaded['model_file'] = loaded['model_file'].replace('~', home)
        full_dcy = loaded['params']['dcys']['model_dcy'].replace('~', home)
        loaded['params']['dcys']['model_dcy'] = full_dcy
        jm_ = JetModel.load_model(loaded["model_file"])
        params = loaded["params"]

        if 'log' in loaded:
            new_modelrun = cls.__init__(jm_, params, log=loaded['log'])
        else:
            dcy = os.path.dirname(os.path.expanduser(loaded['model_file']))
            log_file = os.sep.join([dcy,
                                    os.path.basename(load_file).split('.')[0]
                                    + '.log'])
            new_modelrun = cls.__init__(jm_, params, log=logger.Log(log_file))

        new_modelrun.runs = loaded["runs"]

        return new_modelrun

    @staticmethod
    def py_to_dict(py_file):
        """
        Convert .py file (full path as str) containing relevant model parameters
        to dict
        """
        if not os.path.exists(py_file):
            raise FileNotFoundError(py_file + " does not exist")
        if os.path.dirname(py_file) not in sys.path:
            sys.path.append(os.path.dirname(py_file))

        pl_ = __import__(os.path.basename(py_file)[:-3])
        err = miscf.check_pline_params(pl_.params)

        if err:
            raise err

        sys.path.remove(os.path.dirname(py_file))

        # if not os.path.exists(py_file):
        #     raise FileNotFoundError(py_file + " does not exist")
        # if os.path.dirname(py_file) not in sys.path:
        #     sys.path.append(os.path.dirname(py_file))
        #
        # jp = __import__(os.path.basename(py_file).rstrip('.py'))
        # err = miscf.check_pline_params(jp.params)
        # if err is not None:
        #     raise err

        return pl_.params

    def __init__(self, jetmodel: JetModel, params: Union[dict, str],
                 log: Union[None, logger.Log] = None):
        """

        Parameters
        ----------
        jetmodel : JetModel
            Instance of JetModel to work with
        params : dict or str
            Either a dictionary containing all necessary radiative transfer and
            synthetic observation parameters, or a full path to a parameter
            file.
        log
            logger.Log instance acting as a log. Default is None.
        """
        import time

        # Check validity of args
        if isinstance(jetmodel, JetModel):
            self.model = jetmodel
        else:
            raise TypeError("Supplied arg jetmodel must be JetModel instance "
                            "not {}".format(type(jetmodel)))

        if isinstance(params, dict):
            err = miscf.check_pline_params(params)
            if err:
                raise err
            self._params = params
        elif isinstance(params, str):
            self._params = Pipeline.py_to_dict(params)
        else:
            raise TypeError("Supplied arg params must be dict or full path ("
                            "str)")

        self.dcy = self.params['dcys']['model_dcy'].rstrip(os.sep)
        self.model_file = self.dcy + os.sep + "jetmodel.save"
        self.save_file = self.dcy + os.sep + "pipeline.save"
        self.ptgfile = self.dcy + os.sep + 'pointings.ptg'

        # Create working directory and log
        log_name = "Pipeline_{}.log".format(time.strftime("%Y%m%d%H-%M-%S",
                                                          time.localtime()))
        if not os.path.exists(self.dcy):
            os.mkdir(self.dcy)
            if log is not None:
                self._log = log
            else:
                self._log = logger.Log(fname=os.sep.join([self.dcy, log_name]))
            self.log.add_entry(mtype="INFO",
                               entry=f"Creating pipeline directory, {self.dcy}")
        else:
            if log is not None:
                self._log = log
            else:
                self._log = logger.Log(fname=os.sep.join([self.dcy, log_name]))

        # Make sure that Pipeline and JetModel logs are the same object
        if self.model.log is None:
            self.model.log = self.log
        else:
            new_log = logger.Log.combine_logs(self.log, self.model.log,
                                              self.log.filename,
                                              delete_old_logs=True)
            self.log = self.model.log = new_log

        # Sort Continuum/RRL runs into time order
        if self.params['continuum']['times'] is not None:
            self.params['continuum']['times'].sort()
        else:
            self.params['continuum']['times'] = np.array([])

        if self.params['rrls']['times'] is not None:
            self.params['rrls']['times'].sort()
        else:
            self.params['rrls']['times'] = np.array([])

        # Determine continuum and RRL RT/SO runs to be conducted
        runs = []

        # Determine continuum run parameters
        # cparams = miscf.standardise_pline_params(self.params['continuum'])
        t_obs = self.params['continuum']['t_obs']
        tscps = self.params['continuum']['tscps']
        t_ints = self.params['continuum']['t_ints']
        bws = self.params['continuum']['bws']
        chanws = self.params['continuum']['chanws']
        self.log.add_entry(mtype="INFO",
                           entry="Reading continuum runs into pipeline")
        idx1, idx2 = None, None
        for idx1, t in enumerate(self.params['continuum']['times']):
            for idx2, freq in enumerate(self.params['continuum']['freqs']):
                run = ContinuumRun(
                    self.dcy, t, freq,
                    bws[idx2] if miscf.is_iter(bws) else bws,
                    chanws[idx2] if miscf.is_iter(chanws) else chanws,
                    t_obs[idx2] if miscf.is_iter(t_obs) else t_obs,
                    t_ints[idx2] if miscf.is_iter(t_ints) else t_ints,
                    tscps[idx2] if miscf.is_iter(tscps) else tscps
                )
                runs.append(run)
        if idx1 is None and idx2 is None:
            self.log.add_entry(mtype="WARNING", entry="No continuum runs found",
                               timestamp=True)

        t_obs = self.params['rrls']['t_obs']
        tscps = self.params['rrls']['tscps']
        t_ints = self.params['rrls']['t_ints']
        bws = self.params['rrls']['bws']
        chanws = self.params['rrls']['chanws']
        self.log.add_entry(mtype="INFO",
                           entry="Reading radio recombination line runs into "
                                 "pipeline")
        idx1, idx2 = None, None
        for idx1, t in enumerate(self.params['rrls']['times']):
            for idx2, line in enumerate(self.params['rrls']['lines']):
                run = RRLRun(self.dcy, t, line,
                             bws[idx2] if miscf.is_iter(bws) else bws,
                             chanws[idx2] if miscf.is_iter(chanws) else chanws,
                             t_obs[idx2] if miscf.is_iter(t_obs) else t_obs,
                             t_ints[idx2] if miscf.is_iter(t_ints) else t_ints,
                             tscps[idx2] if miscf.is_iter(tscps) else tscps)
                runs.append(run)

        if idx1 is None and idx2 is None:
            self.log.add_entry(mtype="WARNING", entry="No RRL runs found",
                               timestamp=True)

        self._runs = runs
        self.log.add_entry(mtype="INFO",
                           entry=self.__str__(), timestamp=True)

    def __str__(self):
        hdr = ['Year', 'Type', 'Telescope', 't_obs', 't_int', 'Line',
               'Frequency', 'Bandwidth', 'Channel width',
               'Radiative Transfer?', 'Synthetic Obs.?', 'Completed?']
        units = ['yr', '', '', 's', 's', '', 'Hz', 'Hz', 'Hz', '', '', '']
        fmt = ['.2f', '', '', '.0f', '.0f', '', '.3e', '.3e', '.3e', '', '', '']
        vals = []

        for run in self.runs:
            val = [run.year, run.obs_type.capitalize(), run.tscop,
                   run.t_obs, run.t_int,
                   None if run.obs_type == 'continuum' else run.line,
                   run.freq, run.bandwidth, run.chanwidth,
                   run.radiative_transfer, run.simobserve, run.completed]

            for i, v in enumerate(val):
                if v is None:
                    val[i] = '-'

            vals.append(val)

        tab_head = []
        for i, h in enumerate(hdr):
            if units[i] != '':
                tab_head.append(h + '\n[' + units[i] + ']')
            else:
                tab_head.append(h)

        tab = tabulate.tabulate(vals, tab_head, tablefmt="psql", floatfmt=fmt,
                                numalign='center', stralign='center')

        return tab

    def save(self, save_file: str, absolute_directories: bool = False):
        """
        Saves the pipeline state as a pickled file

        Parameters
        ----------
        save_file: str
            Full path for pickle to save pipeline state to
        absolute_directories: bool
            Save all paths as absolute (system dependent) paths? e.g. the
            home directory prefix to a directory will be stored as '~' if set to
            False. This allows saves to be relevant on multiple
            systems which share a common directory strutuce through
            file sharing, for example. Default is False.

        Returns
        -------
        None
        """
        home = os.path.expanduser('~')
        rs = self.runs
        for idx, run in enumerate(rs):
            # for key in run:
            if not absolute_directories:  # and type(run[key]) is str:
                rs[idx].dcy = run.dcy.replace(home, '~')
            # rs[idx] = run

        ps = self._params
        mf = self.model_file

        if not absolute_directories:
            ps['dcys']['model_dcy'] = ps['dcys']['model_dcy'].replace(home, '~')
            mf = mf.replace(home, '~')

        p = {"runs": rs,
             "params": ps,
             "model_file": mf,
             'log': self.log}

        self.log.add_entry(mtype="INFO",
                           entry="Saving pipeline to " + save_file)
        pickle.dump(p, open(save_file, 'wb'))

        return None

    @property
    def params(self):
        return self._params

    @property
    def dcy(self):
        return self._dcy

    @dcy.setter
    def dcy(self, path):
        self._dcy = path

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

    @log.setter
    def log(self, new_log):
        self._log = new_log

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
        from datetime import datetime, timedelta
        import RaJePy.casa as casa
        import RaJePy.casa.tasks as tasks
        import RaJePy.maths as maths

        self.log.add_entry("INFO", "Beginning pipeline execution")
        if verbose != self.log.verbose:
            self.log.verbose = verbose

        # Target coordinates as SkyCoord instance
        tgt_c = SkyCoord(self.model.params['target']['ra'],
                         self.model.params['target']['dec'],
                         unit=(u.hourangle, u.degree), frame='fk5')

        if simobserve:
            # Make pointing file
            ptg_txt = "#Epoch     RA          DEC      TIME(optional)\n"
            ptg_txt += f"J2000 {tgt_c.to_string('hmsdms')} "

            self.log.add_entry("INFO",
                               "Creating pointings and writing to file, "
                               f"{self.ptgfile}, for synthetic observations")
            with open(self.ptgfile, 'wt') as f:
                f.write(ptg_txt)

        if resume:
            if os.path.exists(self.model_file):
                self.model = JetModel.load_model(self.model_file)

        pfunc.geometry_plot(self.model, show_plot=False,
                            savefig=os.sep.join([self.dcy,
                                                 'GridPlot.pdf']))

        pfunc.jml_profile_plot(self, show_plot=False,
                               savefig=os.sep.join([self.dcy, 'JMLPlot.pdf']))

        for idx, run in enumerate(self.runs):
            self.model.time = run.year * con.year
            self.log.add_entry(mtype="INFO",
                               entry="Executing run #{} -> Details:\n{}"
                                     "".format(idx + 1, run.__str__()))
            if run.completed and resume and not clobber:
                self.log.add_entry(mtype="INFO",
                                   entry="Run #{} previously completed, "
                                         "skipping".format(idx + 1, ),
                                   timestamp=False)
                continue
            try:
                # Create relevant directories
                if not os.path.exists(run.rt_dcy):
                    self.log.add_entry(mtype="INFO",
                                       entry="{} doesn't exist, "
                                             "creating".format(run.rt_dcy),
                                       timestamp=False)
                    # if not dryrun:
                    os.makedirs(run.rt_dcy)

                # Plot physical jet model, if required
                model_plotfile = os.sep.join([os.path.dirname(run.rt_dcy),
                                              "ModelPlot.pdf"])
                if not os.path.exists(model_plotfile) or clobber:
                    pfunc.model_plot(self.model, savefig=model_plotfile,
                                     show_plot=False)

                if not dryrun and run.radiative_transfer:
                    self.log.add_entry(mtype="INFO",
                                       entry="Conducting radiative transfer at "
                                             f"{run.freq / 1e9:.1f}GHz for a "
                                             f"model time of {run.year:.1f}yr")

                    # Compute Emission measures for model plots
                    if not os.path.exists(run.fits_em) or clobber:
                        self.log.add_entry(mtype="INFO",
                                           entry="Emission measures saved to "
                                                 f"{run.fits_em}")
                        self.model.emission_measure(savefits=run.fits_em)
                    else:
                        self.log.add_entry(mtype="INFO",
                                           entry="Emission measures already "
                                                 f"exist -> {run.fits_em}",
                                           timestamp=False)

                    # Radiative transfer
                    if run.obs_type == 'continuum':
                        if not os.path.exists(run.fits_tau) or clobber:
                            self.log.add_entry(mtype="INFO",
                                               entry="Computing optical depths "
                                                     "and saving to "
                                                     f"{run.fits_tau}")
                            self.model.optical_depth_ff(run.chan_freqs,
                                                        savefits=run.fits_tau)
                        else:
                            self.log.add_entry(mtype="INFO",
                                               entry="Optical depths already "
                                                     f"exist -> {run.fits_tau}",
                                               timestamp=False)
                        if not os.path.exists(run.fits_flux) or clobber:
                            self.log.add_entry(mtype="INFO",
                                               entry="Calculating fluxes and "
                                                     "saving to "
                                                     f"{run.fits_flux}")
                            fluxes = self.model.flux_ff(run.chan_freqs,
                                                        savefits=run.fits_flux)
                        else:
                            self.log.add_entry(mtype="INFO",
                                               entry="Fluxes already exist -> "
                                                     f"{run.fits_flux}",
                                               timestamp=False)
                            fluxes = fits.open(run.fits_flux)[0].data
                    else:
                        if not os.path.exists(run.fits_tau) or clobber:
                            self.log.add_entry(mtype="INFO",
                                               entry="Computing optical depths "
                                                     "and saving to "
                                                     f"{run.fits_tau}")
                            self.model.optical_depth_rrl(run.line,
                                                         run.chan_freqs,
                                                         savefits=run.fits_tau)
                        else:
                            self.log.add_entry(mtype="INFO",
                                               entry="Optical depths already "
                                                     f"exist -> {run.fits_tau}",
                                               timestamp=False)
                        if not os.path.exists(run.fits_flux) or clobber:
                            self.log.add_entry(mtype="INFO",
                                               entry="Calculating fluxes and "
                                                     "saving to "
                                                     f"{run.fits_flux}")
                            fluxes = self.model.flux_rrl(run.line,
                                                         run.chan_freqs,
                                                         contsub=False,
                                                         savefits=run.fits_flux)
                        else:
                            self.log.add_entry(mtype="INFO",
                                               entry="Fluxes already exist -> "
                                                     f"{run.fits_flux}",
                                               timestamp=False)
                            fluxes = fits.open(run.fits_flux)[0].data

                    if run.obs_type == 'continuum':
                        # For continuum, flux calculated as average sum flux of
                        # each channel
                        flux = np.nansum(np.nanmean(fluxes, axis=0))
                        self.log.add_entry(mtype="INFO",
                                           entry="Total, average, channel flux "
                                                 f"of {flux:.2e}Jy calculated")
                    else:
                        # For RRL, flux calculated per channel and given as list
                        # of length equal to the number of frequency channels
                        flux = np.nansum(np.nansum(fluxes, axis=1), axis=1)
                    self.runs[idx].results['flux'] = flux

                    # Save model data if doesn't exist
                    if not os.path.exists(self.model_file):
                        self.model.save(self.model_file)

                    # Save pipeline state after successful run
                    self.save(self.save_file, absolute_directories=True)

            except KeyboardInterrupt:
                self.log.add_entry("ERROR",
                                   "Pipeline interrupted by user, saving state")
                self.save(self.save_file)
                self.model.save(self.model_file)
                raise KeyboardInterrupt("Pipeline interrupted by user")

            # Run casa's simobserve and produce visibilities, followed by tclean
            # and then export the images in .fits format
            if simobserve and run.simobserve and not dryrun:
                self.log.add_entry("INFO",
                                   "Setting up synthetic observation CASA "
                                   "script")
                script = casa.Script()
                os.chdir(run.rt_dcy)

                # Get desired telescope name
                tscop, t_cfg = run.tscop

                # Get antennae positions file's path
                ant_list = casa.observatories.tscop_info.cfg_files[tscop][t_cfg]

                # Set frequency (of center channel) and channel width strings by
                # using CASA default parameter values which set the channel
                # width and central channel frequency from the input model
                # header
                chanw_str = ''
                freq_str = ''

                # Get hour-angle ranges above minimum elevation
                min_el = self.params['min_el']
                tscop_lat = casa.observatories.tscop_info.Lat[tscop]

                min_ha = tgt_c.ra.hour - 12.
                if min_ha < 0:
                    min_ha += 24.

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

                # Number of scans through ha-range for E-W interferometers
                # during the final `day of observations'
                # TODO: ARBITRARY HARD-CODED VALUE SET OF 8 SCANS
                ew_split_final_n = 8

                if tscop in casa.observatories.EW_TELESCOPES:
                    ew_int = True

                if ew_int or time_up < run.t_obs:
                    multiple_ms = True

                totaltimes = [time_up] * int(run.t_obs // time_up)
                totaltimes += [run.t_obs - run.t_obs // time_up * time_up]

                self.log.add_entry("INFO",
                                   "Target elevation range of {:+.0f} to "
                                   "{:+.0f}deg with mininum elevation of {}deg "
                                   "and total time on target of {:.1f}hr, means"
                                   " splitting observations over {} run(s)"
                                   "".format(el_range[0], el_range[1], min_el,
                                             run.t_obs / 3600, len(totaltimes)),
                                   timestamp=False)

                # Decide 'dates of observation'
                refdates = []
                for n in range(len(totaltimes)):
                    rdate = (datetime.now() + timedelta(days=n))
                    rdate = rdate.strftime("%Y/%m/%d")
                    refdates.append(rdate)

                # Central hour angles for each observation
                hourangles = ['0h'] * len(totaltimes)

                # If east-west interferometer, spread scans out over range in
                # hour angles
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
                                          skymodel=run.fits_flux,
                                          incenter=freq_str,
                                          inwidth=chanw_str,
                                          setpointings=False,
                                          ptgfile=self.ptgfile,
                                          integration=str(run.t_int) + 's',
                                          antennalist=ant_list,
                                          refdate=refdates[idx2],
                                          hourangle=hourangles[idx2],
                                          totaltime=str(totaltime) + 's',
                                          graphics='none',
                                          overwrite=True,
                                          verbose=True)
                    script.add_task(so)

                # Final measurement set paths
                fnl_clean_ms = run.rt_dcy + os.sep + 'SynObs' + os.sep
                fnl_clean_ms += '.'.join(['SynObs',
                                          os.path.basename(ant_list).rstrip(
                                              '.cfg'),
                                          'ms'])

                fnl_noisy_ms = run.rt_dcy + os.sep + 'SynObs' + os.sep
                fnl_noisy_ms += '.'.join(['SynObs',
                                          os.path.basename(ant_list).rstrip(
                                              '.cfg'),
                                          'noisy', 'ms'])

                if multiple_ms:
                    if os.path.exists(run.rt_dcy + os.sep + 'SynObs'):
                        script.add_task(tasks.Rmdir(
                            path=run.rt_dcy + os.sep + 'SynObs'))
                    script.add_task(tasks.Mkdir(
                        name=run.rt_dcy + os.sep + 'SynObs'))
                    clean_mss, noisy_mss = [], []

                    for project in projects:
                        pdcy = run.rt_dcy + os.sep + project
                        clean_ms = '.'.join([project,
                                             os.path.basename(
                                                 ant_list).rstrip('.cfg'),
                                             'ms'])
                        noisy_ms = '.'.join([project,
                                             os.path.basename(
                                                 ant_list).rstrip('.cfg'),
                                             'noisy', 'ms'])
                        clean_mss.append(pdcy + os.sep + clean_ms)
                        noisy_mss.append(pdcy + os.sep + noisy_ms)

                    script.add_task(tasks.Concat(vis=clean_mss,
                                                 concatvis=fnl_clean_ms))

                    script.add_task(tasks.Concat(vis=noisy_mss,
                                                 concatvis=fnl_noisy_ms))
                    for project in projects:
                        pdcy = run.rt_dcy + os.sep + project
                        script.add_task(tasks.Rmdir(path=pdcy))

                script.add_task(tasks.Chdir(run.rt_dcy + os.sep + 'SynObs'))

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

                max_bl_uvwave = max(ant_pairs.values()) / (con.c / run.freq)
                beam_min = 1. / max_bl_uvwave / con.arcsec

                cell_str = '{:.6f}arcsec'.format(beam_min / 4.)
                cell_size = float('{:.6f}'.format(beam_min / 4.))

                self.log.add_entry("INFO",
                                   "With maximum baseline length of {:.0e}"
                                   " wavelengths, a beam width of {:.2e}arcsec "
                                   "is calculated and therefore using a cell "
                                   "size of {:.2e}arcsec"
                                   "".format(max_bl_uvwave, beam_min,
                                             cell_size), timestamp=False)

                # Define cleaning region as the box encapsulating the flux-model
                # and determine minimum clean image size as twice that of the
                # angular coverage of the flux-model
                ff = fits.open(run.fits_flux)[0]
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
                mod_r_0_au = self.model.params['geometry']['mod_r_0']
                w_0_au = self.model.params['geometry']['w_0']
                tau_0 = mphys.tau_r_from_jm(self.model, run.freq, r_0_au)
                q_tau = self.model.params['power_laws']['q_tau']
                eps = self.model.params['geometry']['epsilon']
                dist_pc = self.model.params['target']['dist']

                jet_deconv_maj_au = (mod_r_0_au * tau_0 ** (-1. / q_tau) +
                                     r_0_au - mod_r_0_au)
                jet_deconv_maj_au *= 2  # For biconical jet
                jet_deconv_maj_as = np.arctan(jet_deconv_maj_au * con.au /
                                              (dist_pc * con.parsec))  # rad
                jet_deconv_maj_as /= con.arcsec  # arcsec

                jet_deconv_min_au = mgeom.w_r(jet_deconv_maj_au / 2.,
                                              w_0_au, mod_r_0_au, r_0_au, eps)
                jet_deconv_min_as = np.arctan(jet_deconv_min_au * con.au /
                                              (dist_pc * con.parsec))  # rad
                jet_deconv_min_as /= con.arcsec  # arcsec

                jet_conv_maj = np.sqrt(jet_deconv_maj_as ** 2. + beam_min ** 2.)
                jet_conv_min = np.sqrt(jet_deconv_min_as ** 2. + beam_min ** 2.)

                if jet_conv_min > jet_conv_maj:
                    jet_conv_maj, jet_conv_min = jet_conv_min, jet_conv_maj

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

                im_name = fnl_noisy_ms.rstrip('ms') + 'imaging'

                if run.obs_type == 'continuum':
                    specmode = 'mfs'
                    restfreq = tasks.Tclean.DEFAULTS['restfreq'][1]
                else:
                    specmode = 'cube'
                    restfreq = [str(run.freq) + 'Hz']

                # Deconvolution of final, noisy, synthetic dataset
                script.add_task(tasks.Tclean(vis=fnl_noisy_ms,
                                             imagename=im_name,
                                             imsize=imsize_cells,
                                             cell=[cell_str],
                                             weighting='briggs',
                                             robust=0.5,
                                             niter=500,
                                             nsigma=3.,
                                             specmode=specmode,
                                             restfreq=restfreq,
                                             mask=mask_str,
                                             interactive=False))

                fitsfile = run.rt_dcy + os.sep + os.path.basename(im_name)
                fitsfile += '.fits'

                script.add_task(tasks.Exportfits(imagename=im_name + '.image',
                                                 fitsimage=fitsfile))

                if run.obs_type == 'continuum':
                    imfit_estimates_file = fitsfile.replace('fits', 'estimates')

                    est_str = '{:.6f}, {:.1f}, {:.1f}, {:.5f}arcsec, ' \
                              '{:.5f}arcsec, ' \
                              '{:.2f}deg'
                    est_str = est_str.format(peak_flux,
                                             imsize_cells[0] / 2.,
                                             imsize_cells[1] / 2.,
                                             jet_conv_maj, jet_conv_min,
                                             self.model.params['geometry'][
                                                 'pa'])

                    with open(imfit_estimates_file, 'wt') as f:
                        f.write(est_str)
                    imfit_results = fitsfile.replace('fits', 'imfit')
                    script.add_task(tasks.Imfit(imagename=fitsfile,
                                                estimates=imfit_estimates_file,
                                                summary=imfit_results))

                self.log.add_entry("INFO", "Executing CASA script {} with a "
                                           "CASA log file, {}"
                                           "".format(script.casafile,
                                                     script.logfile),
                                   timestamp=False)
                script.execute(dcy=run.rt_dcy, dryrun=dryrun)

                if run.obs_type == 'continuum':
                    # noinspection PyUnboundLocalVariable
                    if os.path.exists(imfit_results):
                        run.results['imfit'] = {}
                        with open(imfit_results, 'rt') as f:
                            for idx3, line in enumerate(f.readlines()):
                                if idx3 == 0:
                                    units = [''] + line[1:].split()
                                elif idx3 == 1:
                                    h = line[1:].split()
                                else:
                                    line = [float(_) for _ in line.split()]
                            for idx4, val in enumerate(line):
                                run.results['imfit'][h[idx4]] = {'val': val,
                                                                 'unit': units[
                                                                     idx4]}
                    else:
                        self.log.add_entry("ERROR",
                                           "Run #{}'s imfit failed. Please see "
                                           "casa log, {}, for more details"
                                           "".format(idx + 1,
                                                     run.rt_dcy + os.sep +
                                                     script.casafile))
                        run.results['imfit'] = None

                run.products['ms_noisy'] = fnl_noisy_ms
                run.products['ms_clean'] = fnl_clean_ms
                run.products['clean_image'] = fitsfile
                self.log.add_entry("INFO",
                                   "Run #{}'s synthetic observations completed "
                                   "with noisy measurement set saved to {}, "
                                   "clean measurement set saved to {}, "
                                   "and final, clean image saved to {}"
                                   "".format(idx + 1, fnl_noisy_ms,
                                             fnl_clean_ms, fitsfile))

            self.runs[idx].completed = True

        if not dryrun and simobserve:
            for year in self.params["continuum"]['times']:
                save_file = os.sep.join([self.dcy,
                                         f'RadioSED{year:.1f}yrPlot.png'])
                self.log.add_entry("INFO",
                                   "Saving radio SED figure to "
                                   f"{save_file.replace('png', '(png,pdf)')} "
                                   f"for time {year}yr")
                pfunc.sed_plot(self, year, savefig=save_file)

        self.save(self.save_file)
        self.model.save(self.model_file)

        return None  # self.runs[idx]['products']

    # # TODO: Move this to RaJePy.plotting.functions
    # def plot_continuum_fluxes(self, plot_time: float,
    #                           plot_reynolds: bool = True,
    #                           savefig: bool = False) -> None:
    #     import matplotlib.pylab as plt
    #     freqs, fluxes = [], []
    #     freqs_imfit, fluxes_imfit, efluxes_imfit = [], [], []
    #     for idx, run in enumerate(self.runs):
    #         if run.year == plot_time:
    #             if run.completed and run.obs_type == 'continuum':
    #                 # Skymodel fluxes
    #                 flux = run.results['flux']
    #                 fluxes.append(flux)
    #                 freqs.append(run.freq)
    #
    #                 # imfit fluxes
    #                 if run.results['imfit'] is not None:
    #                     flux_imfit = run.results['imfit']['I']['val']
    #                     eflux_imfit = run.results['imfit']['Ierr']['val']
    #                     fluxes_imfit.append(flux_imfit)
    #                     efluxes_imfit.append(eflux_imfit)
    #                     freqs_imfit.append(run.freq)
    #
    #     freqs = np.array(freqs)
    #     fluxes = np.array(fluxes)
    #
    #     xlims = (10 ** (np.log10(np.min(freqs)) - 0.5),
    #              10 ** (np.log10(np.max(freqs)) + 0.5))
    #
    #     alphas = []
    #     for n in np.arange(1, len(fluxes)):
    #         alphas.append(np.log10(fluxes[n] /
    #                                fluxes[n - 1]) /
    #                       np.log10(freqs[n] / freqs[n - 1]))
    #
    #     alphas_imfit, ealphas_imfit = [], []
    #     for n in np.arange(1, len(fluxes_imfit)):
    #         alphas_imfit.append(np.log10(fluxes_imfit[n] /
    #                                      fluxes_imfit[n - 1]) /
    #                             np.log10(freqs_imfit[n] / freqs_imfit[n - 1]))
    #         c = np.log(freqs_imfit[n] / freqs_imfit[n - 1])
    #         ealpha = np.sqrt((efluxes_imfit[n] / (fluxes_imfit[n] * c)) ** 2. +
    #                          (efluxes_imfit[n - 1] / (
    #                                  fluxes_imfit[n - 1] * c)) ** 2.)
    #         ealphas_imfit.append(ealpha)
    #
    #     l_z = self.model.nz * self.model.csize / \
    #           self.model.params['target']['dist']
    #
    #     plt.close('all')
    #
    #     fig, ax1 = plt.subplots(1, 1, figsize=[cfg.plots["dims"]["column"]] * 2)
    #     ax2 = ax1.twinx()
    #
    #     # Alphas are calculated at the middle of two neighbouring frequencies
    #     # in logarithmic space, hence the need for caclulation of freqs_a,
    #     # the logarithmic mean of the two frequencies
    #     freqs_a = [10. ** np.mean(np.log10([f, freqs[i + 1]])) for i, f in
    #                enumerate(freqs[:-1])]
    #     freqs_a_imfit = [10. ** np.mean(np.log10([f, freqs_imfit[i + 1]])) for
    #                      i, f in
    #                      enumerate(freqs_imfit[:-1])]
    #
    #     ax2.plot(freqs_a, alphas, color='b', ls='None', mec='b', marker='o',
    #              mfc='cornflowerblue', lw=2, zorder=2, markersize=5)
    #
    #     ax2.errorbar(freqs_a, alphas_imfit, yerr=ealphas_imfit, ecolor='b',
    #                  ls='None', capsize=2)
    #
    #     freqs_r86 = np.logspace(np.log10(np.min(xlims)),
    #                             np.log10(np.max(xlims)), 100)
    #     flux_exp = []
    #     for freq in freqs_r86:
    #         f = mphys.flux_expected_r86(self.model, freq, l_z * 0.5)
    #         flux_exp.append(f * 2.)  # for biconical jet
    #
    #     alphas_r86 = []
    #     for n in np.arange(1, len(freqs_r86)):
    #         alphas_r86.append(np.log10(flux_exp[n] / flux_exp[n - 1]) /
    #                           np.log10(freqs_r86[n] / freqs_r86[n - 1]))
    #
    #     # Alphas are calculated at the middle of two neighbouring frequencies
    #     # in logarithmic space, hence the need for caclulation of freqs_a_r86
    #     freqs_a_r86 = [10 ** np.mean(np.log10([f, freqs_r86[i + 1]])) for i, f
    #                    in
    #                    enumerate(freqs_r86[:-1])]
    #     if plot_reynolds:
    #         ax2.plot(freqs_a_r86, alphas_r86, color='cornflowerblue', ls='--',
    #                  lw=2, zorder=1)
    #
    #     ax1.loglog(freqs, fluxes, mec='maroon', ls='None', mfc='r', lw=2,
    #                zorder=3, marker='o', markersize=5)
    #     ax1.errorbar(freqs_imfit, fluxes_imfit, yerr=efluxes_imfit, ecolor='r',
    #                  ls='None', capsize=2)
    #
    #     if plot_reynolds:
    #         ax1.loglog(freqs_r86, flux_exp, color='r', ls='-', lw=2,
    #                    zorder=1)
    #         ax1.loglog(freqs_r86,
    #                    mphys.approx_flux_expected_r86(self.model, freqs_r86) *
    #                    2., color='gray', ls='-.', lw=2, zorder=1)
    #     ax1.set_xlim(xlims)
    #     ax2.set_ylim(-0.2, 2.1)
    #     pfunc.equalise_axes(ax1, fix_x=False)
    #
    #     ax1.set_xlabel(r'$\nu \, \left[ {\rm Hz} \right]$', color='k')
    #     ax1.set_ylabel(r'$S_\nu \, \left[ {\rm Jy} \right]$', color='k')
    #     ax2.set_ylabel(r'$\alpha$', color='b')
    #
    #     ax1.tick_params(which='both', direction='in', top=True)
    #     ax2.tick_params(which='both', direction='in', color='b')
    #     ax2.tick_params(axis='y', which='both', colors='b')
    #     ax2.spines['right'].set_color('b')
    #     ax2.yaxis.label.set_color('b')
    #     ax2.minorticks_on()
    #
    #     save_file = '_'.join(
    #         ["Jet", "lz" + str(self.model.params["grid"]["l_z"]),
    #          "csize" + str(self.model.params["grid"]["c_size"])])
    #     save_file += '.png'
    #     save_file = os.sep.join([self.dcy,
    #                              'Day{}'.format(int(plot_time * 365.)),
    #                              save_file])
    #
    #     title = "Radio SED plot at t={:.0f}yr for jet model '{}'"
    #     title = title.format(plot_time, self.model.params['target']['name'])
    #
    #     png_metadata = cfg.plots['metadata']['png']
    #     png_metadata["Title"] = title
    #
    #     pdf_metadata = cfg.plots['metadata']['pdf']
    #     pdf_metadata["Title"] = title
    #
    #     if savefig:
    #         self.log.add_entry("INFO",
    #                            "Saving radio SED figure to {} for time {}yr"
    #                            "".format(save_file.replace('png', '(png,pdf)'),
    #                                      plot_time))
    #         fig.savefig(save_file, bbox_inches='tight', metadata=png_metadata,
    #                     dpi=300)
    #         fig.savefig(save_file.replace('png', 'pdf'), bbox_inches='tight',
    #                     metadata=pdf_metadata, dpi=300)
    #     return None

    # TODO: Move this to RaJePy.plotting.functions
    def radio_plot(self, run: Union[ContinuumRun, RRLRun],
                   percentile: float = 5., savefig: Union[bool, str] = False):
        """
        Generate 3 subplots of (from left to right) flux, optical depth and
        emission measure.
        
        Parameters
        ----------
        run : ContinuumRun
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
        import matplotlib.pylab as plt
        import matplotlib.gridspec as gridspec

        plt.close('all')

        fig = plt.figure(figsize=(cfg.plots['dims']['text'],
                                  cfg.plots['dims']['column']))

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

        flux = fits.open(run.fits_flux)[0].data[0]
        taus = fits.open(run.fits_tau)[0].data[0]
        ems = fits.open(run.fits_em)[0].data[0]

        flux = np.where(flux > 0, flux, np.NaN)
        taus = np.where(taus > 0, taus, np.NaN)
        ems = np.where(ems > 0, ems, np.NaN)

        # Deal with cube images by averaging along the spectral (1st) axis
        if len(np.shape(flux)) == 3:
            flux = np.nanmean(flux, axis=0)
        if len(np.shape(taus)) == 3:
            taus = np.nanmean(taus, axis=0)

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
        return '{:02.0f}h{:02.0f}m{:06.4f}'.format(h, m, s)

    @property
    def dec(self):
        d = self.coord.dec.dms.d
        m = self.coord.dec.dms.m
        s = self.coord.dec.dms.s
        return '{:+03.0f}d{:02.0f}m{:06.3f}'.format(d, m, s)

    @property
    def duration(self):
        return self._duration

    @property
    def epoch(self):
        return self._epoch

    @property
    def coord(self):
        return self._coord


if __name__ == '__main__':
    # import matplotlib.cm
    # import matplotlib.pylab as plt

    param_dcy = os.sep.join([os.path.dirname(__file__), 'test', 'test_cases'])
    jm = JetModel(os.sep.join([param_dcy, 'test1-model-params.py']))
    pl = Pipeline(jm, os.sep.join([param_dcy, 'test1-pipeline-params.py']))
    cwdcy = os.path.expanduser('~')
    # pl.execute(simobserve=False, resume=False)
    # plotf.plot_geometry(jm, savefig=os.sep.join([cwdcy, 'Desktop',
    #                                              'geometry_plot.pdf']),
    #                     show_plot=False)
    # plotf.model_plot(jm, savefig=os.sep.join([cwdcy, 'Desktop',
    #                                           'model_plot.pdf']),
    #                  show_plot=False)
    # ns = miscf.reorder_axes(jm.number_density, 0, 2, 1, None, 'y', None)
    # hdu3d = fits.PrimaryHDU(ns)
    # hdul3d = fits.HDUList([hdu3d])
    # fitsfile3d = os.sep.join([cwdcy, 'ns3d.fits'])
    # if os.path.exists(fitsfile3d):
    #     os.remove(fitsfile3d)
    # hdul3d.writeto(fitsfile3d)
    # # from RaJePy.plotting import functions as plotf
    # # plotf.model_plot(jm, show_plot=True)
    #
    # ns = jm.fill_factor
    # sum_ns_y = np.nansum(ns, axis=1)
    #
    # if jm._arr_indexing == 'ij':
    #     # hdu = fits.PrimaryHDU(np.flip(np.swapaxes(ns, 2, 0), axis=1))
    #     # Swap x-axis for z-axis and then z-axis for y-axis since
    #     # NAXIS1 = numpy axis 2, NAXIS2 = numpy axis 1 and NAXIS3 = numpy axis 0
    #     hdu3d = fits.PrimaryHDU(np.swapaxes(np.swapaxes(ns, 0, 2), 0, 1))
    #     hdu2d = fits.PrimaryHDU(np.swapaxes(sum_ns_y, 0, 1))
    #     hdu3drs = fits.PrimaryHDU(np.swapaxes(np.swapaxes(jm.rr, 0, 2), 0, 1))
    #     hdu2drs = fits.PrimaryHDU(np.swapaxes(np.nansum(jm.rr, axis=1), 0, 1))
    #     hdu3dphis = fits.PrimaryHDU(np.degrees(np.swapaxes(np.swapaxes(np.where(np.isnan(jm.fill_factor), np.nan, jm.pp), 0, 2), 0, 1)))
    #
    # elif jm._arr_indexing == 'xy':
    #     # hdu = fits.PrimaryHDU(np.nansum(ns, axis=0).T)
    #     hdu3d = fits.PrimaryHDU(ns)
    # else:
    #     raise ValueError(f"Array indexing should be 'ij' or 'xy', not "
    #                      f"{jm._arr_indexing.__repr__()}")
    # hdul3d = fits.HDUList([hdu3d])
    # fitsfile3d = r'C:/Users/simon/Desktop/ns3d.fits'
    # if os.path.exists(fitsfile3d):
    #     os.remove(fitsfile3d)
    # hdul3d.writeto(fitsfile3d)
    #
    # hdul2d = fits.HDUList([hdu2d])
    # fitsfile2d = r'C:/Users/simon/Desktop/ns2d.fits'
    # if os.path.exists(fitsfile2d):
    #     os.remove(fitsfile2d)
    # hdul2d.writeto(fitsfile2d)
    #
    # hdul2drs = fits.HDUList([hdu2drs])
    # fitsfile2drs = r'C:/Users/simon/Desktop/rs2d.fits'
    # if os.path.exists(fitsfile2drs):
    #     os.remove(fitsfile2drs)
    # hdul2drs.writeto(fitsfile2drs)
    #
    # hdul3drs = fits.HDUList([hdu3drs])
    # fitsfile3drs = r'C:/Users/simon/Desktop/rs3d.fits'
    # if os.path.exists(fitsfile3drs):
    #     os.remove(fitsfile3drs)
    # hdul3drs.writeto(fitsfile3drs)
    #
    # hdul3dphis = fits.HDUList([hdu3dphis])
    # fitsfile3dphis = r'C:/Users/simon/Desktop/phis3d.fits'
    # if os.path.exists(fitsfile3dphis):
    #     os.remove(fitsfile3dphis)
    # hdul3dphis.writeto(fitsfile3dphis)
    #
    #
    # print(f"nx, ny, nz = {jm.nx}, {jm.ny}, {jm.nz}")
