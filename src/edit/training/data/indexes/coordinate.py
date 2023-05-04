from __future__ import annotations

import functools
import logging
from typing import Any, Union
import numpy as np
import xarray as xr


from edit.data import DataIndex

from edit.training.data.templates import TrainingOperatorIndex, DataStep
from edit.training.data.sequential import SequentialIterator


@SequentialIterator
class CoordinateIndex(TrainingOperatorIndex):
    """
    OperatorIndex which adds coordinates as data variables from a [Dataset][xarray.Dataset]

    !!! Example
        ```python
        CoordinateIndex(PipelineStep, coordinates = ['latitude'])

        ## As this is decorated with @SequentialIterator, it can be partially initialised

        partialCoordinate = CoordinateIndex(coordinates = ['latitude'])
        partialCoordinate(PipelineStep)
        ```

    """

    def __init__(
        self, index: dict | DataIndex | DataStep, coordinates: str | list[str]
    ):
        """
        Initialise CoordinateIndex

        Will ignore coordinates if they are not in the [Dataset][xarray.Dataset]

        Args:
            index (dict | DataIndex | DataStep):
                Prior Data Retrieval Step, can be dict which will be automatically initialised
            coordinates (str | list[str]):
                Coordinates to add as data variables
        """
        super().__init__(index)

        coordinates = [coordinates] if isinstance(coordinates, str) else coordinates

        self.coordinates = coordinates

    @functools.wraps(DataIndex.get)
    def get(self, query_time: Any) -> xr.Dataset:
        """Retrieve Data at given time, and add coordinates

        Args:
            query_time (Any):
                Time to retrieve data at

        Returns:
            (xr.Dataset):
                Returned [Dataset][xarray.Dataset] with coordinates added
        """
        data = self.index(query_time)
        dims = data.dims

        for coord in self.coordinates:
            if coord in data:
                new_dims = {}
                for key in (key for key in dims.keys() if key not in [coord]):
                    new_dims[key] = np.atleast_1d(data[key].values)

                axis = [list(dims).index(key) for key in new_dims.keys()]
                data[f"var_{coord}"] = data[coord].expand_dims(new_dims, axis=axis)
            else:
                logging.warn(
                    f"{coord} not found in dataset, which has coords: {list(data.coords)}"
                )
        return data

    def _formatted_name(self):
        desc = f"Coordinate Adding Index. Adding {self.coordinates}"
        return super()._formatted_name(desc)
