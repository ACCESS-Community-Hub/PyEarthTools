from abc import abstractmethod
import math
from typing import Union

import numpy as np
import xarray as xr

from edit.training.data.templates import (
    DataOperation,
    DataIterator,
)
from edit.training.data.sequential import Sequential, SequentialIterator


class DataFilter(DataOperation):
    """
    Override __iter__ method to provide a way of filtering the data
    """

    def __init__(self, index) -> None:
        super().__init__(
            index, apply_func=None, undo_func=None, apply_iterator=True, apply_get=False
        )

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError(f"Filter must define Iterator")


@SequentialIterator
class DropNan(DataFilter):
    """
    Drop any data with nans when iterating.
    """

    def _check(self, data: xr.Dataset | np.ndarray):
        if isinstance(data, (xr.Dataset, xr.DataArray)):
            return np.array(list(np.isnan(data).values())).any()
        return np.isnan(data).any()

    def __iter__(self):
        for data in self.index:
            if isinstance(data, tuple):
                if any(tuple(map(self._check, data))):
                    continue
            else:
                if self._check(data):
                    continue
            yield data


@SequentialIterator
class DropAllNan(DataFilter):
    """
    Drop data if it is all nans when iterating.
    """

    def _check(self, data: xr.Dataset | np.ndarray):
        if isinstance(data, (xr.Dataset, xr.DataArray)):
            data = np.isnan(data).all()
            if hasattr(data, "to_array"):
                data = data.to_array()
            return data.values.all()

        elif isinstance(data, xr.DataArray):
            return np.isnan(data).all().values.all()
        return np.isnan(data).all()

    def __iter__(self):
        for data in self.index:
            if isinstance(data, tuple):
                if all(tuple(map(self._check, data))):
                    continue
            else:
                if self._check(data):
                    continue
            yield data


@SequentialIterator
class DropValue(DataFilter):
    """
    Drop Data containing a value above a percentage when iterating.
    """

    def __init__(self, iterator: DataIterator, value: float, percentage: float) -> None:
        """
        Drop Data if number of elements equal to value are greater than percentage when iterating.

        When using __getitem__ do nothing.


        Parameters
        ----------
        iterator
            Iterator
        search_value
            Value to search for
        percentage
            Percentage of which an exceedance drops data
        """
        super().__init__(iterator)

        self.function = (
            lambda x: ((np.count_nonzero(x == value) / math.prod(x.shape)) * 100)
            > percentage
        )
        self.__doc__ = f"Drop data containing more than {percentage}% of {value}"

    def __iter__(self):
        for data in self.index:
            if isinstance(data, tuple):
                if all(self.function(d) for d in data):
                    continue
            else:
                if self.function(data):
                    continue
            yield data
