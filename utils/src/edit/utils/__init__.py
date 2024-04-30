# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

"""
EDIT Utilities
"""

import importlib

## Warnings & Exceptions
from edit.utils.warnings import TesselatorWarning
from edit.utils.exceptions import TesselatorException

## Main imports
from edit.utils import parameter, parsing, repr_utils, imports, context, decorators

## IPython Utils
try:
    from edit.utils.iPython import display_np_arrays_as_images
except ImportError:
    pass


xarray_imported = importlib.util.find_spec("xarray") is not None
if xarray_imported:
    from edit.utils import data

__version__ = "2024.04.01"
