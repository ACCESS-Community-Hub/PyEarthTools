# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
Aggregation based `Modification's`
"""

from __future__ import annotations

from typing import Optional, Union
import xarray as xr

from edit.data.time import EDITDatetime, TimeRange
from edit.data.indexes.utilities.dimensions import identify_time_dimension

from edit.data.modifications import Modification, register_modification


@register_modification("constant")
class Constant(Modification):
    """
    Force a variable to remain constant no matter the time requested.

    Uses `query` if given, otherwise sets it off first time requested.
    Use `memory` to control if precomputed.

    Usage:
    -  !constant[query: '2000-01-01T00', memory: True]:variable
    """

    _data: Optional[xr.Dataset] = None

    def __init__(self, query: Optional[str] = None, memory: Union[str, bool] = True, **kwargs):
        """
        General aggregation

        Args:
            query (Optional[str]):
                Query to use. If None, will use first time retrieved. Defaults to None.
            memory (bool):
                Whether to hold the data in memory. Defaults to True.
        """
        super().__init__(**kwargs)
        self._query = query
        self._memory = memory if isinstance(memory, bool) else memory == "True"

    @property
    def attribute_update(self):
        """Attributes to update on variable"""
        return {"Constant": f"Retrieved from {self._query}"}

    def __repr__(self):
        return f"Constant of {self._variable!r} from {self._query}"

    def single(self, time) -> xr.Dataset:
        self._query = self._query or time

        if self._data is None:
            data = self._data_index(EDITDatetime(self._query).at_resolution(self._data_index.data_resolution))
            if self._memory:
                data = data.compute()

            self._data = data
        assert self._data is not None

        data = self._data
        time_dim = identify_time_dimension(data)
        data = data.assign_coords({time_dim: [time]})
        return data

    def series(self, start, end, interval) -> xr.Dataset:
        data = self.single(start)
        time_dim = identify_time_dimension(data)
        data = data.isel({time_dim: 0})

        data = data.reindex({time_dim: tuple(map(lambda x: x.datetime64, TimeRange(start, end, str(interval))))})
        data = data.ffill(time_dim)
        return data
