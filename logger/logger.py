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
    @classmethod
    def combine_logs(cls, log1: 'Log', log2: 'Log', filename: str,
                     delete_old_logs: bool) -> 'Log':
        """
        Combine two separate logs in to one log with time-sorted entries.
        Writes all previous entries in that order, to new file.

        Parameters
        ----------
        log1 : Log
            First log to combine
        log2 : Log
            Second log to combine
        filename : str
            Full path to new log file
        delete_old_logs : bool
            Whether to delete old log files

        Returns
        -------
        New Log instance.

        """
        # Remove old log files if same as new log file, or if requested
        for logfile in (log1.filename, log2.filename):
            if delete_old_logs or filename == logfile:
                if os.path.exists(logfile):
                    os.remove(logfile)

        # Time sort combined log entries of log1 and log2
        rts = [(log1.entries[k], log1.entries[k].rtime) for k in log1.entries]
        rts += [(log2.entries[k], log2.entries[k].rtime) for k in log2.entries]
        rts = sorted(rts, key=lambda x: x[1])

        all_entries = {n: rts[i][0] for i, n in enumerate(range(len(rts)))}

        # new_log is verbose if either of log1 or log2 is verbose
        new_log = cls(filename, verbose=True in (log1.verbose, log2.verbose))

        # Write all entries from logs to file
        new_log.entries = all_entries
        for n in new_log.entries:
            new_log.write_entry(new_log.entries[n])

        return new_log

    def __init__(self, fname: str, verbose: bool = True):
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

    def add_entry(self, mtype: str, entry: 'Entry',
                  timestamp: bool = True) -> None:
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

        self.write_entry(new_entry)

    def write_entry(self, entry):
        if not os.path.exists(self.filename):
            prefix = ''
        else:
            with open(self.filename, 'rt') as f:
                existing_lines = f.readlines()
                if len(existing_lines) == 0:
                    prefix = ''
                else:
                    prefix = '\n'
        with open(self.filename, 'at+') as f:
            f.write(prefix + entry.__str__())



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

        self._rtime = time.time()  # Time of entry recording (precision)
        self._mtime = time.localtime()  # Time to be displayed in message
        self._mtype = mtype
        self._message = entry
        self.timestamp = timestamp

    def __repr__(self):
        s = "Entry(mtype={}, entry={}, timestamp={})"
        return s.format(self.mtype.__repr__(), self.message.__repr__(),
                        self.timestamp)

    def __str__(self):
        preamble = ':: '.join([self.time_str(),
                               format(self.mtype, str(Entry._mtype_max_len))])

        if not self.timestamp:
            preamble = ' ' * len(preamble)

        fmt_message = self.message.split('\n')
        if len(fmt_message) > 1:
            for i, line in enumerate(fmt_message):
                if i != 0:
                    fmt_message[i] = ' ' * (len(preamble) + 2) + line

        fmt_message = '\n'.join(fmt_message)

        return ': '.join([preamble, fmt_message])
        # else:
        #     return ' ' * (len(preamble) + 2) + ': '.join([self.mtype, self.message])

    @property
    def rtime(self):
        return self._rtime

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

if __name__ == '__main__':
    import numpy as np

    log1 = Log(os.path.expanduser("~") + os.sep + "testlog1.log")
    log2 = Log(os.path.expanduser("~") + os.sep + "testlog2.log")

    for n in range(20):
        if n % 2 == 0:
            log1.add_entry(np.random.choice(Entry._valid_mtypes), str(n))
        else:
            log2.add_entry(np.random.choice(Entry._valid_mtypes), str(n))

    log3 = Log.combine_logs(log1, log2,
                            os.path.expanduser("~") + os.sep + "testlog3.log",
                            True)

    # rtimes = [(log1, log1.entries[k], log1.entries[k].rtime) for k in log1.entries]
    # rtimes += [(log2, log2.entries[k], log2.entries[k].rtime) for k in log2.entries]
    # rtimes = sorted(rtimes, key=lambda x: x[2])
    #
    # all_entries = {n: rtimes[i][1] for i, n in enumerate(range(len(rtimes)))}
    
    # log3 = Log(os.path.expanduser("~") + os.sep + "testlog3.log")
    # log3.entries = all_entries
    # for n in log3.entries:
    #     log3.write_entry(log3.entries[n])
