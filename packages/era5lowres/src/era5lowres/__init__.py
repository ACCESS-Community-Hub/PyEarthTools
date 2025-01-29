# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# ruff: noqa: F401 E402

"""
Index for ERA5 lowres, including downloading helper code and disk indexing for pyearthtools
"""

__version__ = '0.1.0'

import os

import pyearthtools.data
from pyearthtools.data.archive import register_archive

from . import ERA5DataClass

default_base = '/g/data/wb00/NCI-Weatherbench/5.625deg'  # taken from NCI noteboook on github
lowres_base = os.environ.get("ERA5LOWRES", default_base)

ROOT_DIRECTORIES = {
    "era5lowres": lowres_base,  # Update this to the base dir, get var from config
}

# Register archive returns a callable which can be used to register an object
# into the pyearthtools namespace. The root directories of the data set need to be
# registered into the pyearthtools root directories for things to work
register_archive("ROOT_DIRECTORIES")(ROOT_DIRECTORIES)

# Register archive returns a callable which can be used to register an object
# into the pyearthtools namespace. This registered the Python module for the datasets
# into the pyearthtools archives.
register_archive("LOW")(ERA5DataClass)

