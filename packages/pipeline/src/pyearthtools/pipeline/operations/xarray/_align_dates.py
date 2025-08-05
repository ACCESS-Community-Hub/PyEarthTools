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

import pandas as pd
import xarray as xr

from pyearthtools.pipeline.operation import Operation

T = TypeVar("T", xr.Dataset, xr.DataArray)


class AlignDates(Operation):
    """
    Climate datasets often use the cftime module to index into data using non-standard calendars.
    This operation will recode the time coordinate of a dataset or data array to a standard timestamp
    For now, support only exists for recoding from Noleap to Timestamp
    """

    _override_interface = "Serial"

    def __init__(self, to="start_of_month"):
        """
        Record initialisation and store flags for processing

        Args:
            to: either "start_of_month" or a zero-padded string day-of-month
        """

        if to == "start_of_month":
            to = "01"

        if len(to) != 2:
            raise ValueError(f"Value of 'to' {to} is not recognised as a day of the month")
        self.to = to

        super().__init__(
            split_tuples=True,
            operation="apply",
            recognised_types=(xr.Dataset, xr.DataArray),
        )
        self.to = to
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
        to = self.to
        if self.to == "start_of_month":
            to = "01"

        aligned = data.time.dt.strftime(f"%Y-%m-{to}")
        aligned = [pd.Timestamp(a) for a in aligned.values]
        data = data.assign_coords({"time": aligned})

        return data
