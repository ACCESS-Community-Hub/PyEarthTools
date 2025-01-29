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

from __future__ import annotations
from typing import Hashable

import xarray as xr

STANDARD_TIME_COORD_NAMES = ["time", "basetime"]


def identify_time_dimension(data: xr.DataArray | xr.Dataset) -> str:
    """Attempt to identify time dimension in dataset.

    If cannot be identified, return 'time'
    """
    coords = list(str(x) for x in data.coords)

    for coord in coords:
        if coord in STANDARD_TIME_COORD_NAMES:
            return coord

    for coord in coords:
        if "time" in coord:
            return coord

    for coord in coords:
        dtype = data[coord].dtype
        if "time" in dtype.__class__.__name__:
            return coord

    return "time"
