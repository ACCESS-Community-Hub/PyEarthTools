# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

# ruff: noqa: F401

"""
EDIT Utilities
"""

__version__ = "1.2dev"

import importlib

from edit.utils import parameter, repr_utils, context, decorators, initialisation, config, logger
from edit.utils.initialisation import load, save, dynamic_import


import edit
import importlib.util

xarray_imported = importlib.util.find_spec("xarray") is not None
if xarray_imported:
    from edit.utils import data

setattr(edit, "config", config)

