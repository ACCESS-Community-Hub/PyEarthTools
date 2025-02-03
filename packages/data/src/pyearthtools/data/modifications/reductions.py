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
Reduction based `Modification's`
"""

from __future__ import annotations

from typing import Union
import xarray as xr

from pyearthtools.data.time import pyearthtoolsDatetime, TimeDelta, TimeResolution
from pyearthtools.data.indexes.utilities.dimensions import identify_time_dimension

from pyearthtools.data.modifications import Modification, register_modification


class Reduction(Modification):
    pass
    # def __call__(self, dataset: xr.Dataset, variable: str) -> xr.DataArray:
    #     modified_variable = super().__call__(dataset, variable)
    #     time_dim = identify_time_dimension(dataset)
    #     # print(f"Reduction/\n", dataset, modified_variable)

    #     if modified_variable[time_dim].equals(dataset[time_dim]):
    #         return modified_variable
    #     return modified_variable


class Groupby(Reduction):
    def __init__(self, time_component: str, method: str, **kwargs):
        super().__init__(**kwargs)
        self.time_component = time_component
        self.method = method

    def _reconstruct_time_dim(self, time: Union[str, pyearthtoolsDatetime], new_coord: xr.DataArray):
        time = pyearthtoolsDatetime(time)

        return list(
            map(
                lambda x: x.at_resolution(self.time_component).datetime64("ns"),
                [time + (TimeDelta(i, self.time_component)) for i in range(len(new_coord))],
            )
        )

    def _get_data(self, start, end):
        dataset = self._data_index.series(start, end, inclusive=True)
        time_dim = identify_time_dimension(dataset)

        grouped_data = dataset.groupby(f"{time_dim}.{self.time_component}")
        grouped_data = getattr(grouped_data, self.method)()

        # grouped_data = grouped_data.rename({self.time_component: time_dim})
        return grouped_data.assign(
            {self.time_component: self._reconstruct_time_dim(start, grouped_data[self.time_component])}
        )

    def single(self, time) -> xr.Dataset:
        start = str(pyearthtoolsDatetime(time).at_resolution(TimeResolution(self.time_component)))
        end = str(pyearthtoolsDatetime(time).at_resolution(TimeResolution(self.time_component)) + 1)
        return self._get_data(start, end)

    def series(self, start, end, interval) -> xr.Dataset:
        start = str(pyearthtoolsDatetime(start).at_resolution(TimeResolution(self.time_component)))
        end = str(pyearthtoolsDatetime(end).at_resolution(TimeResolution(self.time_component)) + 1)

        return self._get_data(start, end)


@register_modification("hourly")
def Hourly(method: str = "mean", **kwargs):
    return Groupby(time_component="hour", method=method, **kwargs)


@register_modification("daily")
def Daily(method: str = "mean", **kwargs):
    return Groupby(time_component="day", method=method, **kwargs)


@register_modification("monthly")
def Monthly(method: str = "mean", **kwargs):
    return Groupby(time_component="month", method=method, **kwargs)
