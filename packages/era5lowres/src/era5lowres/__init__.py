# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

# ruff: noqa: F401 E402

"""
Index for ERA5 lowres, including downloading helper code and disk indexing for EDIT
"""

__version__ = '0.1.dev1'

import os

import edit.data
from edit.data.archive import register_archive

from . import ERA5DataClass

default_base = '/g/data/wb00/NCI-Weatherbench/5.625deg'  # taken from NCI noteboook on github
lowres_base = os.environ.get("ERA5LOWRES", default_base)

ROOT_DIRECTORIES = {
    "era5lowres": lowres_base,  # Update this to the base dir, get var from config
}

# Register archive returns a callable which can be used to register an object
# into the EDIT namespace. The root directories of the data set need to be
# registered into the EDIT root directories for things to work
register_archive("ROOT_DIRECTORIES")(ROOT_DIRECTORIES)

# Register archive returns a callable which can be used to register an object
# into the EDIT namespace. This registered the Python module for the datasets
# into the EDIT archives.
register_archive("LOW")(ERA5DataClass)

