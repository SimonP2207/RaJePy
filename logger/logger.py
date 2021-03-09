# -*- coding: utf-8 -*-
"""
Classes associated with the creation, keeping and editing of log entries for a
model run.
"""
import os
import errno
import time
import traceback


class Log:
    """
    Class to handle creation, modification and storage of log entries
    """

    def __init__(self, fname, verbose=True):
        """
        Parameters
        ----------
        fname : str
            Full path to log file
        verbose : bool
            Whether to print log entries verbosely.
        """
        self._entries = {}
        self._filename = fname
        self._verbose = verbose

    def __str__(self):
        es = []
        for entry_num in range(1, len(self.entries) + 1):
            es.append(self.entries[entry_num].__str__())

        return '\n'.join(es)

    @property
    def filename(self):
        return self._filename

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, new_verbosity):
        self._verbose = new_verbosity

    @property
    def entries(self):
        return self._entries

    @entries.setter
    def entries(self, new_entries):
        self._entries = new_entries

    def add_entry(self, mtype, entry, timestamp=True):
        """
        Add entry to log

        Parameters
        ----------
        mtype : str
            Log entry type (info, error or warning only)
        entry: str
            Message to enter into log
        timestamp: bool
            Whether to include the timestamp in the log entry
        Returns
        -------
        None.

        """
        if not os.path.exists(os.path.dirname(self.filename)):
            # Raise FileNotFoundError (subclass of builtin OSError) correctly
            raise FileNotFoundError(errno.ENOTDIR,
                                    os.strerror(errno.ENOTDIR),
                                    os.path.dirname(self.filename))

        if not os.path.exists(self.filename):
            # os.mknod(self.filename) --> fails with PermissionError
            open(self.filename, 'w').close()

        new_entry = Entry(mtype, entry, timestamp)
        new_entries = self.entries
        new_entries[len(self._entries) + 1] = new_entry
        self.entries = new_entries

        if self.verbose:
            print(new_entry)

        with open(self.filename, 'at') as f:
            f.write(('\n' if len(self.entries) != 1 else "") +
                    new_entry.__str__())


class Entry:
    """
    Entry class for use with Log class
    """
    _valid_mtypes = ("INFO", "ERROR", "WARNING")
    _mtype_max_len = max([len(_) for _ in _valid_mtypes])

    @classmethod
    def valid_mtypes(cls):
        return cls._valid_mtypes

    @classmethod
    def mtype_max_len(cls):
        return cls._mtype_max_len

    def __init__(self, mtype: str, entry: str,# calling_obj: str,
                 timestamp: bool = True):
        """
        Parameters
        ----------
        mtype : str
            Message type. One of 'INFO', 'ERROR' or 'WARNING' (any case)
        entry: str
            Entry message
        calling_obj: str
            Object name instantiating the Entry
        timestamp: bool
            Whether to include the timestamp in the log entry string
        Returns
        -------
        None.

        """
        if not isinstance(mtype, str):
            raise TypeError("mtype must be a str")

        if not isinstance(entry, str):
            raise TypeError("entry must be a str")

        if mtype.upper() not in Entry.valid_mtypes():
            raise TypeError("mtype must be one of '" +
                            "', '".join(self._valid_mtypes[:-1]) + "' or '" +
                            self._valid_mtypes[-1] + "'")

        self._mtime = time.localtime()
        self._mtype = mtype
        self._message = entry
        self.timestamp = timestamp

    def __repr__(self):
        s = "Entry(mtype={}, entry={}, timestamp={})"
        return s.format(self.mtype.__repr__(), self.message.__repr__(),
                        self.timestamp)

    def __str__(self):
        if self.timestamp:
            return ':: '.join([self.time_str(), self.mtype, self.message])
        else:
            return ' ' * (len(self.time_str()) + 3) + ':: '.join([self.mtype,
                                                                  self.message])

    @property
    def message(self):
        return self._message
    
    @property
    def mtype(self):
        return self._mtype

    @property
    def mtime(self):
        return self._mtime

    def time_str(self, fmt='%d%B%Y-%H:%M:%S'):
        return time.strftime(fmt, self.mtime).upper()
