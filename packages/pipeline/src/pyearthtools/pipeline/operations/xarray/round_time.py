# Copyright Commonwealth of Australia, Bureau of Meteorology 2025.
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

from typing import Optional, Callable

import pandas as pd
import xarray as xr

from pyearthtools.pipeline.operation import Operation


class RoundTime(Operation):
    """Round the time coordinate to the nearst hour."""

    _override_interface = "Serial"

    def __init__(
        self,
        freq: str,
        fun: Optional[Callable | str] = None
    ):
        """
        Round timestamps to specified frequency resolution. For example time values
        like 00:30UTC become 01:00UTC.
        Wrapper of xarray.DataArray.dt.round

        Args:
            freq (str):
                a freq string indicating the rounding resolution e.g. “D” for daily resolution
            fun (Callable, optional):
                function to use when rounding. Default is 'round'
            
        """
        super().__init__(
            split_tuples=True,
            operation="apply",
            recognised_types=(xr.Dataset, xr.DataArray),
        )
        
        self._freq = freq
        self._fun = fun
        self._default_fun = "round"
        self._known_methods = ["round", "ceil", "floor"]
        self._known_freq = ['H', 'D']

        if self._fun is None:
            self._fun = self._default_fun

        if not self._fun in self._known_methods:
            raise KeyError(f"Rounding function of {self._fun} not found in {self._known_methods}")

        if not self._freq in self._known_freq:
            raise KeyError(f"freq of {self._freq} not implemented! Only {self._known_freq} accepted!")

        self.record_initialisation()

    def apply_func(self, dataset: xr.Dataset) -> xr.Dataset:
        # Get the function from the dataset directly
        _fun = getattr(dataset['time'].dt, self._fun)

        # Get the new time steps and make sure they are pandas timestamps
        newtime = _fun(self._freq)
        newtime = [pd.Timestamp(a) for a in newtime.values]

        # Assign the new time values to the time coordinate
        dataset = dataset.assign_coords({"time": newtime})

        return dataset