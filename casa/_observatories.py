"""
Script to take the output of casa interface commands:

tablename = os.sep.join([os.getenv("CASAPATH").split(' ')[0], 'data',
                         'geodetic', 'Observatories'])
tb.open(tablename)
tb.toasciifmt('Observatories_table.txt')
tb.close()

Which is the file, 'Observatories_table.txt', and create a pandas dataframe
from the outputted text file.
"""
import os
import numpy as np
import pandas as pd
import RaJePy._config as config

fname = 'Observatories_table.txt'
fname = os.sep.join([os.path.expanduser('~'), 'Nextcloud', 'PythonLibraries',
                     'VaJePy', 'files', 'antenna_configs', fname])

data = np.empty((0, 12))
with open(fname, 'rt') as f:
    for idx, line in enumerate(f.readlines()):
        str_begin = False
        breaks = [0]
        for idx2, char in enumerate(line):
            if char == ' ' and not str_begin:
                breaks.append(idx2)
            elif char == '"':
                if not str_begin:
                    str_begin = True
                else:
                    str_begin = False

        l = [line[i:j] for i, j in zip(breaks, breaks[1:] + [None])]
        l = [_.strip().strip('"') for _ in l]
        data = np.append(data, np.array([l]), axis=0)

dtypes = data[1]
tscop_info = pd.DataFrame(data=data[2:],  # values
                          index=data[2:, 1],  # 1st column as index
                          columns=data[0])  # 1st row as the column names

for idx, col in enumerate(tscop_info.columns):
    if dtypes[idx] == 'D':
        tscop_info[col] = tscop_info[col].astype('float')
    else:
        tscop_info[col] = tscop_info[col].astype('str')

tscop_info['cfg_files'] = None
tscop_info = tscop_info.applymap(lambda x: {} if x is None else x)

# Telescope latitudes, longitudes and antenna configuration files
# ACA/ALMA
tscop_info['cfg_files']['ACA']['0'] = 'aca.cfg'
tscop_info['cfg_files']['ALMA']['C-1'] = 'alma.C-1.cfg'
tscop_info['cfg_files']['ALMA']['C-2'] = 'alma.C-2.cfg'
tscop_info['cfg_files']['ALMA']['C-3'] = 'alma.C-3.cfg'
tscop_info['cfg_files']['ALMA']['C-4'] = 'alma.C-4.cfg'
tscop_info['cfg_files']['ALMA']['C-5'] = 'alma.C-5.cfg'
tscop_info['cfg_files']['ALMA']['C-6'] = 'alma.C-6.cfg'
tscop_info['cfg_files']['ALMA']['C-7'] = 'alma.C-7.cfg'
tscop_info['cfg_files']['ALMA']['C-8'] = 'alma.C-8.cfg'
tscop_info['cfg_files']['ALMA']['C-9'] = 'alma.C-9.cfg'
tscop_info['cfg_files']['ALMA']['C-10'] = 'alma.C-10.cfg'

# ATCA
tscop_info['cfg_files']['ATCA']['6A'] = 'atca_6a.cfg'
tscop_info['cfg_files']['ATCA']['6B'] = 'atca_6b.cfg'
tscop_info['cfg_files']['ATCA']['6C'] = 'atca_6c.cfg'
tscop_info['cfg_files']['ATCA']['6D'] = 'atca_6d.cfg'

# EMERLIN
tscop_info['cfg_files']['EMERLIN']['0'] = 'emerlin.cfg'

# GMRT
tscop_info['cfg_files']['GMRT']['0'] = 'gmrt.cfg'

# LOFAR
tscop_info['cfg_files']['LOFAR']['0'] = 'LOFAR.cfg'

# MeerKAT
tscop_info['cfg_files']['MeerKAT']['0'] = 'meerkat.cfg'

# NGVLA
tscop_info['cfg_files']['NGVLA']['SBA'] = 'ngvla-sba-revC.cfg'
tscop_info['cfg_files']['NGVLA']['MIDSUBARRAY'] = 'ngvla-mid-subarray-revC.cfg'
tscop_info['cfg_files']['NGVLA']['CORE'] = 'ngvla-core-revC.cfg'
tscop_info['cfg_files']['NGVLA']['PLAINS'] = 'ngvla-plains-revC.cfg'
tscop_info['cfg_files']['NGVLA']['MAIN'] = 'ngvla-main-revC.cfg'
tscop_info['cfg_files']['NGVLA']['FULL'] = 'ngvla-revC.cfg'
tscop_info['cfg_files']['NGVLA']['LBA'] = 'ngvla-lba-revC.cfg'

# PdBI
tscop_info['cfg_files']['IRAM_PDB']['A'] = 'pdbi-a.cfg'
tscop_info['cfg_files']['IRAM_PDB']['B'] = 'pdbi-b.cfg'
tscop_info['cfg_files']['IRAM_PDB']['C'] = 'pdbi-c.cfg'
tscop_info['cfg_files']['IRAM_PDB']['D'] = 'pdbi-d.cfg'

# SMA
tscop_info['cfg_files']['SMA']['SUBCOMPACT'] = 'sma.subcompact.cfg'
tscop_info['cfg_files']['SMA']['COMPACT'] = 'sma.compact.cfg'
tscop_info['cfg_files']['SMA']['EXTENDED'] = 'sma.extended.cfg'
tscop_info['cfg_files']['SMA']['VEXTENDED'] = 'sma.vextended.cfg'

# VLA
tscop_info['cfg_files']['VLA']['A'] = 'vla.a.cfg'
tscop_info['cfg_files']['VLA']['BnA'] = 'vla.bna.cfg'
tscop_info['cfg_files']['VLA']['B'] = 'vla.a.cfg'
tscop_info['cfg_files']['VLA']['CnB'] = 'vla.cnb.cfg'
tscop_info['cfg_files']['VLA']['C'] = 'vla.a.cfg'
tscop_info['cfg_files']['VLA']['DnC'] = 'vla.dnc.cfg'
tscop_info['cfg_files']['VLA']['D'] = 'vla.a.cfg'

# VLBA
tscop_info['cfg_files']['VLBA']['0'] = 'vlba.cfg'

# WSRT
tscop_info['cfg_files']['WSRT']['0'] = 'wsrt.cfg'

# Add full parent directorys' paths for configuration files before filenames
for tscop in tscop_info.Name:
    if tscop_info.loc[tscop].cfg_files == {}:
        pass
    else:
        for cfg in tscop_info.loc[tscop].cfg_files:
            full_path = os.sep.join([config.dcys["files"],
                                     'antenna_configs',
                                     tscop_info['cfg_files'][tscop][cfg]])
            tscop_info.cfg_files[tscop][cfg] = full_path
