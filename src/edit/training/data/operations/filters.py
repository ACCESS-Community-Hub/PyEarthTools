from __future__ import annotations

from abc import abstractmethod
import math

import numpy as np
import xarray as xr

from edit.training.data.templates import (
    DataOperation,
    DataIterator,
    DataStep,
)
from edit.training.data.sequential import Sequential, SequentialIterator


class DataFilter(DataOperation):
    """
    DataOperation Child to override `__iter__` method to provide a way of filtering data
    Parent Class of Data Filters

    !!! Warning
        Cannot be used directly, this is simply the parent class to provide a structure

    !!! Example
        ```python
        DataFilterChild(PipelineStep)

        ## As this is decorated with @SequentialIterator, it can be partially initialised

        partialDataFilterChild = DataFilterChild()
        partialDataFilterChild(PipelineStep)
        ```
    """
    def __init__(self, index : DataStep) -> None:
        """DataOperation to filter incoming data         
        
        Args:
            index (DataStep): 
                Underlying DataStep to get data from
        """        
        super().__init__(
            index, apply_func=None, undo_func=None, apply_iterator=True, apply_get=False
        )

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError(f"Child Filter must define Iterator")

    def _sanity(self):
        return 'Filter'

@SequentialIterator
class DropNan(DataFilter):
    """
    DataFilter to drop any data with nans when iterating.
    """

    def _check(self, data: xr.Dataset | np.ndarray) -> bool:
        """Check if any of the data is nan        
        
        Args:
            data (xr.Dataset | np.ndarray): 
                Data to check
        
        Returns:
            (bool): 
                If data contains nan's
        """        
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
    DataFilter to drop data if it is all nans when iterating.
    """

    def _check(self, data: xr.Dataset | np.ndarray) -> bool:
        """Check if all of the data is nan        
        
        Args:
            data (xr.Dataset | np.ndarray): 
                Data to check
        
        Returns:
            (bool): 
        """      
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
    DataFilter to drop data containing more than given percentage of a value.
    """
    def __init__(self, iterator: DataStep, value: float, percentage: float) -> None:
        """Drop Data if number of elements equal to value are greater than percentage when iterating.        
        
        Args:
            iterator (DataStep): 
                Underlying DataStep to get data from
            value (float): 
                Value to search for
            percentage (float): 
                Percentage of `value` of which an exceedance drops data
        """        
        super().__init__(iterator)

        self.function = (
            lambda x: ((np.count_nonzero(x == value) / math.prod(x.shape)) * 100)
            >= percentage
        )

        self.__doc__ = f"Drop data containing more than {percentage}% of {value}."
        self._info_ = dict(value = value, percentage = percentage)

    def __iter__(self):
        for data in self.index:
            if isinstance(data, tuple):
                if all(self.function(d) for d in data):
                    continue
            else:
                if self.function(data):
                    continue
            yield data
