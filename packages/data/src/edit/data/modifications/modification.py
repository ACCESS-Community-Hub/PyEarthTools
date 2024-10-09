# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
Base Class for Modification's
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod

from typing import Any, Union
import xarray as xr

import edit.data
from edit.data.indexes import TimeDataIndex
from edit.data.indexes.utilities.dimensions import identify_time_dimension

from edit.data.transforms.transform import TransformCollection


class Modification(metaclass=ABCMeta):
    """
    Modifications to `variables` for Data Indexes

    These are to be used when modifying variables at a core level, such that the more information is needed
    then what is returned upon a simple index into the data.

    For example:
        Creating an accumulation:
            When getting data at particular time step, an accumulation cannot be found as it requires
            prior information, a `modification` can then go and get this to create the accumulation.

            This is how it differs from a `transform`, as they transform the data retrieved, and this
            creates and modifies effectively as it is being retrieved.

    Implementing:
        To implement a Modification `single` & `series` must be provided.

        `single` takes a single timestep and expects a dataset to be returned with the variable as modified.

        `series` takes a start, end and interval, as can be parsed by `edit.data.TimeRange`, and expects
        a dataset to be returned with the variable as modified but all timesteps as defined by the range.

        `variable` contains the variable being modified.

        `data` contains the `TimeDataIndex` to retrieve the data from.

        `attribute_update` can be overridden to specify a dictionary to update the attributes with.

    """

    _data_index: TimeDataIndex  # Underlying data class

    def __init__(
        self,
        variable: str,
        index_class: TimeDataIndex,
        index_kwargs: dict[str, Any],
        variable_keyword: str,
    ):
        """
        Setup Modification

        Args:
            variable (str):
                Variable being modified
            index_class (TimeDataIndex):
                Class where data is being sourced from
            index_kwargs (dict[str, Any]):
                Kwargs used to init `index_class`
            variable_keyword (str):
                Keyword for `variable` when initing `index_class`
        """
        index_kwargs.pop("self", None)
        index_kwargs.pop(variable_keyword, None)

        self._data_index = index_class.__class__(**{variable_keyword: variable}, **index_kwargs)
        self._data_index.base_transforms = TransformCollection(index_class.base_transforms[:-1])
        self._variable = variable

    @property
    def attribute_update(self):
        """Attributes to update on variable"""
        return {}

    @property
    def data(self):
        """Get the `TimeDataIndex` as specified by the user in which to find the modification."""
        return self._data_index

    @property
    def variable(self):
        """Variable being modified as given by the user."""
        return self._variable

    def __repr__(self):
        return f"{self.__class__.__name__} on {self._variable}"

    @abstractmethod
    def single(self, time) -> xr.Dataset:
        """Get the modification for a single timestep"""
        raise NotImplementedError

    @abstractmethod
    def series(self, start, end, interval) -> xr.Dataset:
        """Get the modification for a series of timesteps"""
        raise NotImplementedError

    def __call__(self, dataset: xr.Dataset, variable: str) -> Union[xr.DataArray, xr.Dataset]:
        """
        Get the modification

        Args:
            dataset (xr.Dataset):
                Template of data, used to determine the time being operated on
            variable (str):
                Variable name of the modified data

        Returns:
            (xr.DataArray):
                Newly created and modified data
        """
        time_dim = identify_time_dimension(dataset)
        time_coord = dataset[time_dim]

        if len(time_coord) == 1:
            data = self.single(time_coord.values[0])
        else:
            time_values = time_coord.values
            start_time = time_values[0]
            end_time = time_values[-1]

            interval = (end_time - start_time) / len(time_values)
            data = self.series(start_time, end_time, interval)

        if isinstance(data, xr.Dataset):
            data = data[variable]

        return edit.data.transforms.attributes.update(self.attribute_update)(data)
