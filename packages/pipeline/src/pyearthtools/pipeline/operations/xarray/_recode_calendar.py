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


from typing import TypeVar

import xarray as xr

from pyearthtools.pipeline.operation import Operation

T = TypeVar("T", xr.Dataset, xr.DataArray)


class RecodeCalendar(Operation):
    """
    Climate datasets often use the cftime module to index into data using non-standard calendars.
    This operation will recode the time coordinate of a dataset or data array to a standard timestamp
    For now, support only exists for recoding from Noleap to Timestamp
    """

    _override_interface = "Serial"

    def __init__(self):
        """
        Record initialisation and store flags for processing
        """

        super().__init__(
            split_tuples=True,
            operation="apply",
            recognised_types=(xr.Dataset, xr.DataArray),
        )
        self.record_initialisation()

    def apply_func(self, data: xr.Dataset) -> xr.Dataset:
        """Sort an `xarray` object data variables into the given order

        Args:
            data (T):
                `xarray` object to sort.

        Returns:
            (T):
                Sorted dataset
        """

        recoded = data.indexes["time"].to_datetimeindex()
        data["time"] = recoded

        return data
