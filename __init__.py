from . import _config as cfg
from . import _constants as cnsts
from .classes import *
from . import logger
from . import maths
from . import plotting
from . import casa
from . import miscellaneous

# Numerical version:
__version_info__ = (0, 0, 2)
__version__ = '.'.join(map(str, __version_info__))

__author__ = 'Simon Purser <simonp2207@gmail.com>'
