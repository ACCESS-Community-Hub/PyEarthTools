import math
from typing import Union

import numpy as np
import xarray as xr

from dset.training.data.templates import (
    DataIterationOperator,
    DataIterator,
    SequentialIterator,
)

@SequentialIterator
class DropNan(DataIterationOperator):
    """
    Drop any data with nans when iterating.
    """

    def _check(self, data: Union[xr.Dataset, np.array]):
        if isinstance(data, (xr.Dataset, xr.DataArray)):
            return np.array(list(np.isnan(data).values())).any()
        return np.isnan(data).any()

    def __iter__(self):
        for data in self.iterator:
            if isinstance(data, tuple):
                if any(tuple(map(self._check, data))):
                    continue
            else:
                if self._check(data):
                    continue
            yield data


@SequentialIterator
class DropAllNan(DataIterationOperator):
    """
    Drop data if it is all nans when iterating.
    """

    def _check(self, data: Union[xr.Dataset, np.array]):
        if isinstance(data, (xr.Dataset, xr.DataArray)):
            return np.array(list(np.isnan(data).values())).all()
        return np.isnan(data).all()

    def __iter__(self):
        for data in self.iterator:
            if isinstance(data, tuple):
                if all(tuple(map(self._check, data))):
                    continue
            else:
                if self._check(data):
                    continue
            yield data


@SequentialIterator
class DropValue(DataIterationOperator):
    """
    Drop Data containing a value above a percentage
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
        for data in self.iterator:
            if isinstance(data, tuple):
                if all(self.function(d) for d in data):
                    continue
            else:
                if self.function(data):
                    continue
            yield data
