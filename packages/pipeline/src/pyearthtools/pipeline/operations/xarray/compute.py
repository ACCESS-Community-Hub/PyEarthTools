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


from typing import TypeVar

import xarray as xr

from pyearthtools.pipeline.operation import Operation

T = TypeVar("T", xr.Dataset, xr.DataArray)


class Compute(Operation):
    """Compute xarray object"""

    _override_interface = "Serial"

    def __init__(self):
        super().__init__(
            split_tuples=True,
            recognised_types=(xr.Dataset, xr.DataArray),
            operation="apply",
        )
        self.record_initialisation()

    def apply_func(self, sample: T) -> T:
        if hasattr(sample, "compute"):
            return sample.compute()
        return sample
