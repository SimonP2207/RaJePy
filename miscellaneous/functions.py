from collections.abc import Iterable
from typing import Union
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

def is_float(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

def casa_imfit_file_to_dict(filename):
    """
    Convert CASA's imfit output file (defined by its 'summary' parameter) to
    a python dict
    Parameters
    ----------
    filename : `str`
        Full path to text file containing imfit results

    Returns
    -------
    return_dict : `dict`
        Python dict format of imfit results
    """
    data = []
    with open(filename, 'rt') as f:
        for idx, line in enumerate(f.readlines()):
            if idx in (0, 1):
                line = line.strip('#')
            line = [_.strip() for _ in line.split()]
            line = [float(_) if is_float(_) else _ for _ in line]
            if idx == 0:
                line.insert(0, '')
            data.append(line)

    return_dict = {}
    for i, param in enumerate(data[1]):
        return_dict[param] = {'units': data[0][i], 'value': data[2][i]}

    return return_dict


def _param_key_check(params, keys):
    for section in keys:
        if section not in params:
            return KeyError("{} keyword not found in params "
                            "dict".format(section))
        if not isinstance(keys[section], Iterable):
            if not isinstance(params[section], keys[section]):
                return ValueError("value of {} section of "
                                  "params must be of type {}, "
                                  "not {}".format(section, keys[section],
                                                  type(params[section])))
            continue

        for line in keys[section]:
            key, typ = line[0], line[1]

            if key not in params[section]:
                return KeyError("{} keyword not found in {} section of params "
                                "dict".format(key, section))

            val = params[section][key]
            if not isinstance(typ, (Iterable, type(None))):
                if not isinstance(val, (typ, type(None))):
                    return ValueError("{} value of {} section of "
                                      "params must be of type {}, "
                                      "not {}".format(key, section,
                                                      typ, type(val)))
            else:
                if not isinstance(val, (Iterable, type(None))):
                    return ValueError("{} value of {} section of "
                                      "params must be of type {}, "
                                      "not {}".format(key, section,
                                                      typ[0],
                                                      type(val)))
                if not isinstance(val, type(None)):
                    if len(val) != 0 and not isinstance(val[0], typ[1]):
                        return ValueError("{} of params's section {}'s value, "
                                          "{}, must contain objects of type "
                                          "{}, not {}".format(typ[0],
                                                              section, key,
                                                              typ[1],
                                                              type(val[0])))

    return None


def check_pline_params(params):
    if not isinstance(params, dict):
        return TypeError("model params must be dict")

    keys = {'min_el': float,
            'dcys': (('model_dcy', str),),
            'continuum': (('times', (np.ndarray, np.float)),
                          ('freqs', (np.ndarray, np.float)),
                          ('t_obs', (np.ndarray, np.integer)),
                          ('tscps', (np.ndarray, np.ndarray)),
                          ('t_ints', (np.ndarray, np.integer)),
                          ('bws', (np.ndarray, np.float)),
                          ('chanws', (np.ndarray, np.float))),
            'rrls': (('times', (np.ndarray, np.float)),
                     ('lines', (np.ndarray, np.str)),
                     ('t_obs', (np.ndarray, np.integer)),
                     ('tscps', (np.ndarray, np.ndarray)),
                     ('t_ints', (np.ndarray, np.integer)),
                     ('bws', (np.ndarray, np.float)),
                     ('chanws', (np.ndarray, np.float)))
            }

    e = _param_key_check(params, keys)
    if isinstance(e, Exception):
        return e

    # Extra, pipeline-specific checks here
    for band in ('continuum', 'rrls'):
        shape = np.shape(params[band]['tscps'])
        if shape != (0, ) and shape != ():
            if shape[1] is not 2:
                return ValueError("np.ndarray of params's section {}'s value, "
                                  "tscps, must be of shape (n, 2)".format(band))


def check_model_params(params):
    if not isinstance(params, dict):
        return TypeError("model params must be dict")

    keys = {'target': (('name', str),
                       ('ra', str),
                       ('dec', str),
                       ('epoch', str),
                       ('dist', float),
                       ('v_lsr', float),
                       ('M_star', float),
                       ('R_1', float),
                       ('R_2', float)),
            'grid': (('n_x', int),
                     ('n_y', int),
                     ('n_z', int),
                     ('l_z', float),
                     ('c_size', float)),
            'geometry': (('epsilon', float),
                         ('opang', float),
                         ('w_0', float),
                         ('r_0', float),
                         ('inc', float),
                         ('pa', float),
                         ('mod_r_0', float)),
            'power_laws': (('q_v', float),
                           ('q_T', float),
                           ('q_x', float),
                           ('q_n', float),
                           ('q_tau', float),
                           ('q^d_n', float),
                           ('q^d_T', float),
                           ('q^d_v', float),
                           ('q^d_x', float)),
            'properties': (('v_0', float),
                           ('x_0', float),
                           ('n_0', float),
                           ('T_0', float),
                           ('mu', float),
                           ('mlr', float)),
            'ejection': (('t_0', (np.ndarray, np.float)),
                         ('hl', (np.ndarray, np.float)),
                         ('chi', (np.ndarray, np.float)))}

    e = _param_key_check(params, keys)
    if isinstance(e, Exception):
        return e

    # Extra, model-specific checks here
    try:
        if params['target']['epoch'] == 'J2000':
            frame = 'fk5'
        elif params['target']['epoch'] == 'B1950':
            frame = 'fk4'
        else:
            return ValueError("Only epochs B1950 and J2000 are supported as "
                              "values for epoch within model parameters' "
                              "target specifications")
        SkyCoord(params["target"]["ra"], params["target"]["dec"],
                 frame=frame, unit=(u.hourangle, u.degree))
    except ValueError:
        return ValueError("Please check validity of sexagesimal coordinates "
                          "within ra/dec fields of target section of model "
                          "params, as well as a valid value for frame")

def freq_str(freq: Union[Iterable, float],
             fmt: str = '.0f') -> Union[Iterable, float]:
    """
    Return string of a frequency in sensible units

    Parameters
    ---------
    freq: Iterable, float
        Frequency(s) of which to format
    fmt: str
        Accuracy/format of returned frequency string(s)

    Returns
    -------
    String or list of strings representing input frequencies with units
    """
    suffixes = {'Hz': {'min_freq': 1., 'max_freq': 1e3},
                'kHz': {'min_freq': 1e3, 'max_freq': 1e6},
                'MHz': {'min_freq': 1e6, 'max_freq': 1e9},
                'GHz': {'min_freq': 1e9, 'max_freq': 1e12},
                'THz': {'min_freq': 1e12, 'max_freq': 1e15},
                'PHz': {'min_freq': 1e15, 'max_freq': 1e18},}

    def find_suffix(f):
        for suffix in suffixes:
            d = suffixes[suffix]
            if d['min_freq'] <= f and d['max_freq'] > f:
                return suffix

    if not isinstance(freq, Iterable):
        suffix = find_suffix(freq)
        fac = suffixes[suffix]['min_freq']
        return f'{{:{fmt}}}{{}}'.format(freq / fac, suffix)

    else:
        freq_strs = []
        for f in freq:
            suffix = find_suffix(f)
            fac = suffixes[suffix]['min_freq']
            freq_strs.append(f'{{:{fmt}}}{{}}'.format(f / fac, suffix))
        return freq_strs

def is_iter(x):
    return isinstance(x, Iterable)

if __name__ == '__main__':
    import os
    imfit_file = os.sep.join([os.path.expanduser('~'), 'Dropbox',
                              'Paper_RadioRT', 'Results', 'FluxLossModel1',
                              'Day0', '10GHz',
                              'SynObs.vla.a.noisy.imaging.imfit'])
    a = casa_imfit_file_to_dict(imfit_file)
