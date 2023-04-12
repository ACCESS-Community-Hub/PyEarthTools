
from typing import Any, Union
import xarray as xr
import numpy as np
import datetime


import dset.data
from dset.data import archive, transform, TransformCollection, dset_datetime
from dset.data.default import OperatorIndex

from dset.training.data.utils import get_transforms
from dset.training.data.templates import SequentialIterator, TrainingOperatorIndex

@SequentialIterator
class CoordinateIndex(TrainingOperatorIndex):
    """
    Add Coordinates as data variable
    """
    def __init__(self, 
        index: Union[dict, OperatorIndex],
        coordinates: Union[str, list[str]]
        ):
        """
        Create CoordinateAdder

        Parameters
        ----------
        index
            Dictionary with keys as imports or modules to other OperatorIndexes
        coordinates:
            Coordinates to add
        """
        super().__init__(index)

        coordinates = [coordinates] if isinstance(coordinates, str) else coordinates

        self.coordinates = coordinates
        
    def get(self, query_time):
        data = self.index[query_time]
        dims = data.dims

        for coord in self.coordinates:
            if coord in data:
                new_dims = {}
                for key in (key for key in dims.keys() if key not in [coord]):
                    new_dims[key] = np.atleast_1d(data[key].values)
                
                axis = [list(dims).index(key) for key in new_dims.keys()]
                data[f"var_{coord}"] = data[coord].expand_dims(new_dims,axis=axis)
            else:
                raise KeyError(f"{coord} not found in dataset, which has coords: {list(data.coords)}")
        return data

    def _formatted_name(self):
        desc = f"Coordinate Adding Index. Adding {self.coordinates}"
        return super()._formatted_name(desc)

