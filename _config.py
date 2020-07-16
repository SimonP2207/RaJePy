"""
RaJePy configuration file.

Purpose:
    - Defines the locations of VaJePy libraries and data files.
    - Defines plot dimensions
"""
import os

dcys = {"scripts": os.path.dirname(os.path.realpath(__file__)),
        "files": os.sep.join([os.path.dirname(os.path.realpath(__file__)),
                              "files"])
        }

plots = {"dims": {"column": 3.32153,  # inches
                  "text": 6.97522  # Inches
                  }
         }
