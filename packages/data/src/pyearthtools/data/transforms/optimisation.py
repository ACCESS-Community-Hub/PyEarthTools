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
from typing import Literal, Any


import xarray as xr
from pyearthtools.data.transforms.transform import Transform

from pyearthtools.utils.decorators import BackwardsCompatibility


class Rechunk(Transform):
    """Rechunk data"""

    def __init__(self, method: int | dict[str, Any] | Literal["auto", "encoding"]):
        """
        Rechunk data

        Args:
            method (int | dict[str, Any] | Literal['auto', 'encoding']):
                Rechunk either by encoding, auto or by variable config.
        """
        if not isinstance(method, (int, dict)) and method not in ["auto", "encoding"] and method is not None:
            raise ValueError(f"method must be an int, dict, 'auto', 'encoding', or None. Instead found {method}.")
        self._method = method

    def apply(self, dataset: xr.Dataset) -> xr.Dataset:
        for var in dataset:
            chunks = self._method
            if chunks == "encoding":
                if "chunksizes" in dataset[var].encoding:
                    chunks = dataset[var].encoding["chunksizes"] or "auto"
                else:
                    raise ValueError(f"Could not find 'chunksizes' in encoding of {var}")
            dataset[var].data = dataset[var].data.rechunk(chunks)
        return dataset


@BackwardsCompatibility(Rechunk)
def rechunk(*args, **kwargs): ...
