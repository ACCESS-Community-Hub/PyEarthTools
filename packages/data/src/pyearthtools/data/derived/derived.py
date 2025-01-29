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
Derived Data
"""

from __future__ import annotations

from typing import Union
import inspect
from abc import abstractmethod, ABCMeta

import xarray as xr


from pyearthtools.data.time import pyearthtoolsDatetime, TimeDelta, TimeRange
from pyearthtools.data.indexes import DataIndex, TimeDataIndex, AdvancedTimeDataIndex


class DerivedValue(DataIndex, metaclass=ABCMeta):
    """Base class for Derived data

    Subclassed from `DataIndex` so transforms can be used.

    Child must implement `derive`.
    """

    @abstractmethod
    def derive(self, *args, **kwargs) -> xr.Dataset:
        """
        Get derived value.

        Will only be passed most specific key, so if a function of time, expect a time.

        Child class must implement
        """
        ...

    def get(self, *args, **kwargs):
        """Override for get to use `derive`."""
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, pyearthtoolsDatetime):
                args[i] = arg.datetime64()
        return self.derive(*args, **kwargs)

    @classmethod
    def like(cls, dataset: Union[xr.Dataset, xr.DataArray], **kwargs):
        """
        Setup DerivedValue taking coords from `dataset` if key in `__init__`.

        If `cls` takes `latitude` and `longitude`, and those coords in `dataset`, will take `values`, and pass
        to `__init__`

        Examples:
        ```python
        era = pyearthtools.data.archive.ERA5.sample()
        derived = DerivedValue.like(era['2000-01-01T00'])
        ```
        """
        init_parameters = inspect.signature(cls.__init__).parameters

        init_values = {}
        for key in set(init_parameters.keys()).intersection(dataset.coords).difference(kwargs.keys()):
            init_values[key] = dataset.coords[key].values

        return cls(**kwargs, **init_values)


class TimeDerivedValue(DerivedValue, TimeDataIndex):
    """
    Temporally derived value Index

    """

    def __init__(self, data_interval: tuple[int, str] | int | str | TimeDelta | None = None, **kwargs):
        """
        Derived value which is a factor of time.

        Hooks into `TimeDataIndex` to allow for series retrieval

        Args:
            data_interval (tuple[int, str] | int | str | TimeDelta | None, optional):
                Default interval of data. Defaults to None.
        """
        super().__init__(data_interval=data_interval, **kwargs)


class AdvancedTimeDerivedValue(TimeDerivedValue, AdvancedTimeDataIndex):
    """
    Advanced Temporally derived value Index.

    Allows for automatic time resolution based retrieval.

    Example
    ```python
        index = AdvancedTimeDerivedValue('6 hours')
        index['2000-01-01'] # Will get four steps 00,06,12,18
    ```
    """

    def __init__(
        self, data_interval: tuple[int, str] | int | str | TimeDelta | None = None, split_time: bool = False, **kwargs
    ):
        """
        Advanced Temporally Derived Index

        Args:
            data_interval (tuple[int, str] | int | str | TimeDelta | None, optional):
                Interval of derivation, if given allows for [] to get multiple samples based on resolution. Defaults to None.
            split_time (bool, optional):
                Whether to split a series call into each individual time, or pass list of times. Defaults to False.
        """
        super().__init__(data_interval, **kwargs)
        self._split_time = split_time

    def series(
        self,
        start,
        end,
        interval: TimeDelta | tuple[int | float, str] | int | None = None,
        **_,
    ):
        if not self._split_time:
            return self.derive(list(map(lambda x: x.datetime64(), TimeRange(start, end, self._get_interval(interval)))))

        return xr.combine_by_coords(
            tuple(self.derive(x.datetime64()) for x in TimeRange(start, end, self._get_interval(interval)))
        )

    def __dir__(self):
        dir = list(super().__dir__())
        for method in ["aggregation", "range", "safe_series"]:
            dir.remove(method)
        return dir
