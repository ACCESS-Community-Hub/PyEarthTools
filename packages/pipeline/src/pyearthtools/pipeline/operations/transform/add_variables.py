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


"""
Add variables
"""
from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr
from pyearthtools.data import Transform


class TimeOfYear(Transform):
    """
    Add time of year to dataset

    """

    def __init__(self, method: str):
        """
        Add time of year as variable to a dataset

        Use [DropDataset][pyearthtools.pipeline.operations.select.DropDataset] to remove it
        if an earlier step in the pipeline is sensitive to variable names.

        Args:
            method (str):
                Method to use, either "dayofyear" or "monthofyear"
                Both modelled as a sinusodal function

        Returns:
            (Transform):
                Transform to add time of year variable
        """
        super().__init__()
        self.record_initialisation()

        if method not in ["dayofyear", "monthofyear"]:
            raise ValueError(f"Invalid method passed, cannot be {method!r}. Must be in ['dayofyear', 'monthofyear']")
        self.method = method

    def apply(self, ds: xr.Dataset):
        dims = ds.dims

        if self.method == "dayofyear":
            value = (np.cos(ds.time.dt.dayofyear * np.pi / (366 / 2)) + 1) / 2
        if self.method == "monthofyear":
            value = (np.cos(ds.time.dt.date.month * np.pi / 6) + 1) / 2

        new_dims = {}

        for key in (key for key in dims.keys() if key not in ["time"]):
            new_dims[key] = np.atleast_1d(ds[key].values)

        axis = [list(dims).index(key) for key in new_dims.keys()]
        ds[self.method] = value.expand_dims(new_dims, axis=axis)

        # value = value * np.ones([len(ds[dim]) for dim in list(ds.dims)])
        # ds[method] = (ds.dims, value)

        dims = ds[list(ds.data_vars)[0]].dims

        ds = ds.transpose(*list(dims))
        return ds

    @property
    def _info_(self) -> Any | dict:
        return dict(method=self.method)
