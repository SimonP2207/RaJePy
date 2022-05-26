"""
Contains any classes responsible for the scripting of casa tasks and pipelines
which are subsequently executed on the command line using 'casa -c [script]'
"""

import os


class Script(object):
    """
    Class to handle a collection of coherent list of casa tasks/tool and execute
    that collection with casa, in the order in which it is given.
    """
    def __init__(self):
        self._tasklist = []

        # Must always add e-MERLIN's primary beam response to CASA's vpmanager
        from RaJePy.casa.tasks import AddGaussPBresponse
        from datetime import datetime as dt

        fwhm_str = '{:.3f}deg'.format(1.71768e10 / (1e9 * 25.))
        maxrad_str = '{:.3f}deg'.format(3.43537e10 / (1e9 * 25.))

        self.add_task(AddGaussPBresponse(telescope='MERLIN2',
                                         halfwidth=fwhm_str,
                                         maxrad=maxrad_str,
                                         reffreq='1GHz'))

        prefix = dt.now().strftime("%d%m%Y_%H%M%S")
        self._logfile = prefix + '.log'
        self._casafile = prefix + '.py'

    @property
    def tasklist(self):
        return self._tasklist

    @tasklist.setter
    def tasklist(self, new_tasklist):
        self._tasklist = new_tasklist

    def add_task(self, new_task):
        from collections.abc import Iterable
        if not isinstance(new_task, Iterable):
            self.tasklist.append(new_task)
        else:
            for task in new_task:
                self.tasklist.append(task)

    @property
    def logfile(self):
        return self._logfile

    @property
    def casafile(self):
        return self._casafile

    def execute(self, dcy=os.getcwd(), dryrun=False):
        import subprocess

        if dcy != os.getcwd():
            os.chdir(dcy)

        with open(dcy + os.sep + self.casafile, 'a+') as lf:
            # Necessary imports within CASA environment
            lf.write('import os\nimport shutil\n')
            for task in self.tasklist:
                lf.write(str(task) + '\n')

        cmd = "casa --nogui --nologger --agg --logfile '{}' -c '{}'"

        if dryrun:
            print(cmd.format(dcy + os.sep + self.logfile,
                             dcy + os.sep + self.casafile))
            print("Contents of {}:".format(dcy + os.sep + self.casafile))
            with open(dcy + os.sep + self.casafile, 'rt') as lf: print(
                lf.read())

        else:
            op = subprocess.run(cmd.format(dcy + os.sep + self.logfile,
                                           dcy + os.sep + self.casafile),
                                shell=True)
