#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Handles all casa task wrappers (if that's the correct word!)
@author: simon
"""
import os

import numpy as np


class _CasaTask(object):
    """
    Parent class to handle CASA tasks
    """

    def __init__(self, taskname, taskparameters, *args, **kwargs):
        """

        Parameters
        ----------
        taskname : str
            Name of the task e.g. 'tclean'
        taskparameters : dict
            Dictionary containing task's parameter names as keys whose values
            are the 2-tuples of the data type expected of that parameter and
            its default value e.g.:
                taskparameters = {'imagename': (str, ''),
                                  'flux': (float, 0.0),
                                  'index': (int, 0),
                              s    ...}
        **kwargs : passed down to the set_val method. Kwarg keys should be valid
            parameter names for the casa task with their values being the valid
            type for that parameter.
        """
        # Check correct data types in tasks' supplied parameters
        task_params, exp_params = {}, {}
        for key in taskparameters:
            p = taskparameters[key]

            if type(key) not in (int, str):
                raise TypeError('Expected parameters\' keys must be str or int '
                                'types, not {} ({})'.format(type(key), key))
            if type(p[0]) not in (type, tuple):
                raise TypeError('Data type or N-tuple of types must be first '
                                'value in parameter {}s 2-tuple '
                                'value not {}'.format(key, str(p[0])))
            if type(p[0]) is tuple:
                if set([type(_) for _ in (float, int)]) != {type}:
                    raise TypeError('Tuple must only contain types not '
                                    '{}'.format([type(_) for _ in p[0]]))
                if type(p[1]) not in p[0]:
                    raise TypeError("Data-value's type must be one of " +
                                    str(p[0]) + " not " + str(type(p[1])) +
                                    ", for " + key)
            elif type(p[1]) is not p[0]:
                raise TypeError("Data-value's type must be " + str(p[0]) +
                                " not " + str(type(p[1])) + ", for " + key)

            task_params[key] = p[1]
            exp_params[key] = p[0]

        self._taskname = taskname
        self._params = task_params
        self._param_types = exp_params
        self._imports = []
        self.set_vals(*args, **kwargs)

    @property
    def imports(self):
        return self._imports

    @imports.setter
    def imports(self, new_imports):
        self.imports = new_imports

    def add_import(self, new_import):
        from collections.abc import Iterable
        if not isinstance(new_import, Iterable):
            self.imports.append(new_import)
        else:
            for i in new_import:
                self.imports.append(i)

    @property
    def taskname(self):
        return self._taskname

    def __str__(self):
        """
        Returns executable-within-casa-environment string for use in
        executable casa scripts. From self.parameters dict, numbered (int
        only) keys will be sorted and placed as *args (not names args), while
        keys that are strs will be placed as **kwargs.

        Returns
        -------
        Executable method within casa-environment e.g.:

            os.chdir('/some/example/path')
            simobserve(project='example_project', skymodel='eg_model', ...)

        """
        def separator(a_list):
            ints, strs = [], []
            for element in a_list:
                if type(element) is int:
                    ints.append(element)
                else:
                    strs.append(element)
            ints.sort()
            strs.sort()

            return ints + strs

        s = '{0}('.format(self.taskname)
        keys = list(self.params.keys())
        keys = separator(keys)

        for idx, key in enumerate(keys):
            if type(key) is str:
                s += key + '='
            if self.param_types[key] is str:
                s += '\"{0}\"'.format(self.params[key])
            else:
                s += '{0}'.format(str(self.params[key]))
            if idx + 1 != len(self.params.keys()):
                s += ', '

        s += ')'
        return s

    @property
    def param_types(self):
        return self._param_types

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, new_params):
        self._params = new_params

    def set_vals(self, *args, **kwargs):
        """

        Parameters
        ----------
        kwargs: Keyword arguments whose key is a valid task parameter and whose
            value is of a valid type for that parameter

        Returns
        -------

        """
        new_ps = self.params

        for idx, arg in enumerate(args):
            new_ps[idx + 1] = arg

        for param in kwargs:
            val = kwargs[param]
            if param not in self.params.keys():
                raise ValueError("{} not a valid parameter for casa task, "
                                 "{}".format(param, self.taskname))
            if type(val) is not self.param_types[param]:
                raise ValueError('supplied val, {}, should be a {} not, '
                                 '{}'.format(val, self.param_types[param],
                                             type(val)))
            new_ps[param] = val
        self.params = new_ps


class Simobserve(_CasaTask):
    """
    Task to handle generation of synthetic measurement sets from a `true'
    model (image)
    """

    def __init__(self, **kwargs):
        ps = {'project': (str, ''),
              'skymodel': (str, ''),
              'incenter': (str, ''),
              'inwidth': (str, ''),
              'complist': (str, ''),
              'setpointings': (bool, False),
              'ptgfile': (str, ''),
              'integration': (str, '5s'),
              'direction': (str, ''),
              'mapsize': (list, ['', '']),
              'maptype': (str, 'ALMA'),
              'pointingspacing': (str, ''),
              'obsmode': (str, 'int'),  # Only 'int' (interferometer) here!
              'antennalist': (str, ''),  # FULL path to antenna config file
              'refdate': (str, ''),
              'hourangle': (str, 'transit'),
              'totaltime': (str, ''),  # e.g. '7200s'
              'caldirection': (str, ''),
              'calflux': (str, '1Jy'),
              'outframe': (str, 'LSRK'),
              'thermalnoise': (str, 'tsys-atm'),
              'user_pwv': (float, 1.0),
              't_ground': (float, 269.0),
              'seed': (int, np.random.randint(1e6, dtype=int)),
              'leakage': (float, 0.0),
              'graphics': (str, 'none'),
              'verbose': (bool, True),
              'overwrite': (bool, False)}
        super().__init__('simobserve', ps, **kwargs)


class Tclean(_CasaTask):
    """
    Task to handle imaging and deconvolution of a measurement set's visibilities
    """

    def __init__(self, **kwargs):
        ps = {'vis': (str, ''),
              'selectdata': (bool, False),
              'field': (str, ''),
              'spw': (str, ''),
              'timerange': (str, ''),
              'uvrange': (str, ''),
              'antenna': (str, ''),
              'scan': (str, ''),
              'observation': (str, ''),
              'intent': (str, ''),
              'datacolumn': (str, 'data'),
              'imagename': (str, ''),
              'imsize': (list, [500, 500]),
              'cell': (list, ['0.1arcsec']),
              'phasecenter': (str, ''),
              'stokes': (str, 'I'),
              'projection': (str, 'TAN'),
              'startmodel': (str, ''),
              'specmode': (str, 'mfs'),
              'restfreq': (list, ['']),
              'reffreq': (str, ''),
              'gridder': (str, 'standard'),
              'vptable': (str, ''),
              'pblimit': (float, 0.2),
              'deconvolver': (str, 'clark'),
              'scales': (list, []),
              'smallscalebias': (float, 0.0),
              'nterms': (int, 1),
              'restoration': (bool, True),
              'restoringbeam': (list, []),
              'pbcor': (bool, False),
              'outlierfile': (str, ''),
              'weighting': (str, 'briggs'),
              'robust': (float, 0.5),
              'noise': (str, '1.0Jy'),
              'npixels': (int, 0),
              'uvtaper': (list, []),
              'niter': (int, 1000),
              'gain': (float, 0.1),
              'threshold': (float, 0.0),
              'nsigma': (float, 3.0),
              'cycleniter': (int, -1),
              'cyclefactor': (float, 1.0),
              'minpsffraction': (float, 0.05),
              'maxpsffraction': (float, 0.8),
              'interactive': (bool, False),
              'usemask': (str, 'user'),
              'mask': (str, ''),
              'pbmask': (float, 0.0),
              'fastnoise': (bool, True),
              'restart': (bool, False),
              'savemodel': (str, 'none'),
              'calcres': (bool, True),
              'calcpsf': (bool, True),
              'parallel': (bool, False)}
        super().__init__('tclean', ps, **kwargs)


class Exportfits(_CasaTask):
    """
    Task to handle conversion of a CASA image to a FITS file
    """

    def __init__(self, **kwargs):
        ps = {'imagename': (str, ''),
              'fitsimage': (str, ''),
              'velocity': (bool, False),
              'optical': (bool, False),
              'bitpix': (int, -32),
              'minpix': (int, 0),
              'maxpix': (int, -1),
              'overwrite': (bool, False),
              'dropstokes': (bool, False),
              'stokeslast': (bool, True),
              'history': (bool, True),
              'dropdeg': (bool, False)}
        super().__init__('exportfits', ps, **kwargs)


class Concat(_CasaTask):
    """
    Task to handle concatenation of measurement sets
    """

    def __init__(self, **kwargs):
        ps = {'vis': (list, ['']),
              'concatvis': (str, ''),
              'freqtol': (str, ''),
              'dirtol': (str, ''),
              'respectname': (bool, False),
              'timesort': (bool, False),
              'copypointing': (bool, True),
              'visweightscale': (list, []),
              'forcesingleephemfield': (str, '')}
        super().__init__('concat', ps, **kwargs)


class Chdir(_CasaTask):
    """
    Change the working directory.

    NOTE that this task does not take kwargs, only a single arg.
    """

    def __init__(self, *args):
        ps = {1: (str, '')}
        super().__init__('os.chdir', ps, *args)


class Mkdir(_CasaTask):
    """
    Change the working directory
    """

    def __init__(self, **kwargs):
        ps = {'name': (str, ''),
              'mode': (int, 0o777),}
              #'exist_ok': (bool, True)}
        super().__init__('os.makedirs', ps, **kwargs)


class Rmdir(_CasaTask):
    """
    Change the working directory
    """

    def __init__(self, **kwargs):
        ps = {'path': (str, ''),
              'ignore_errors': (bool, False)}
        super().__init__('shutil.rmtree', ps, **kwargs)


class Imfit(_CasaTask):
    """
    Perform Gaussian fitting of a 2D flux distribution in the image plane
    """

    def __init__(self, **kwargs):
        ps = {'imagename': (str, ''),
              'box': (str, ''),
              'region': (str, ''),
              'chans': (str, ''),
              'stokes': (str, ''),
              'mask': (str, ''),
              'includepix': (list, []),
              'excludepix': (list, []),
              'residual': (str, ''),
              'model': (str, ''),
              'estimates': (str, ''),
              'logfile': (str, ''),
              'newestimates': (str, ''),
              'complist': (str, ''),
              'dooff': (bool, False),
              'offset': (float, 0.0),
              'fixoffset': (bool, False),
              'rms': ((float, int), -1),
              'noisefwhm': (str, ''),
              'summary': (str, '')}
        super().__init__('imfit', ps, **kwargs)


class Immath(_CasaTask):
    """
    Perform mathematical operations on image data
    """

    def __init__(self, **kwargs):
        ps = {'imagename': (str, ''),
              'mode': (str, 'evalexpr'),
              'expr': (str, ''),
              'varnames': (str, ''),
              'sigma': (str, '0.0mJy/beam'),
              'outfile': (str, 'immath_results.im'),
              'polithresh': (str, ''),
              'mask': (str, ''),
              'region': (str, ''),
              'box': (str, ''),
              'chans': (str, ''),
              'stokes': (str, ''),
              'imagemd': (str, ''),
              'prec': (str, 'float')}
        super().__init__('immath', ps, **kwargs)


class AddGaussPBresponse(_CasaTask):
    """
    Add a telescope to the list of known primary-beam responses via the
    vpmanager and assumption of a Gaussian primary beam
    """
    def __init__(self, **kwargs):
        ps = {'telescope': (str, ''),
              'othertelescope': (str, ''),
              'dopb': (bool, True),
              'halfwidth': (str, '0.5deg'),
              'maxrad': (str, '1.0deg'),
              'reffreq': (str, '1.0GHz'),
              'isthispb': (str, 'PB'),
              'squintdir': (dict, {'m0': {'unit': 'rad', 'value': 0.0},
                                   'm1': {'unit': 'rad', 'value': 0.0},
                                   'refer': 'J2000',
                                   'type': 'direction'}),
              'squintreffreq': (dict,  {'unit': 'GHz', 'value': 1.0}),
              'dosquint': (bool, False),
              'paincrement': (dict, {'unit': 'deg', 'value': 720.0}),
              'usesymmetricbeam': (bool, False)}
        super().__init__('vp.setpbgauss', ps, **kwargs)


# Module testing code below
if __name__ == '__main__':
    from RaJePy.casa import Script

    work_dcy = os.sep.join([os.path.expanduser('~'), 'Desktop', 'testdir'])

    if not os.path.exists(work_dcy):
        os.makedirs(work_dcy)
    os.chdir(work_dcy)

    test_script = Script()
    test_script.add_task(Chdir('/test/directory/for/chdir'))
    test_script.add_task(Simobserve(project="TestProject",
                                    skymodel="test_skymodel.fits",
                                    ptgfile="test_ptgfile.ptg",
                                    antennalist='test_antlist.cfg',
                                    incenter='5GHz',
                                    inwidth='2GHz'))
    test_script.add_task(Tclean(vis='test.ms',
                                imagename='test_imagename',
                                imsize=[1000, 1000],
                                cell=['0.01arcsec'],
                                weighting='briggs',
                                robust=0.5,
                                niter=500))
    test_script.add_task(Exportfits(imagename='test_imagename.image',
                                    fitsimage='test_fits.fits',
                                    dropdeg=True))
    test_script.add_task(Rmdir(path='/Users/simon/test_project_dcy'))
    test_script.execute(dcy=work_dcy, dryrun=True)
