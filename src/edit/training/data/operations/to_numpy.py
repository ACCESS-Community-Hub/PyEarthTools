import functools
import time
from itertools import zip_longest
from typing import Union

import numpy as np
import xarray as xr

from edit.training.data.templates import DataOperation, DataStep
from edit.training.data.iterators.temporal import DataInterface
from edit.training.data.sequential import Sequential, SequentialIterator


@SequentialIterator
class ToNumpy(DataOperation):
    """
    DataOperation to convert data to [np.array][numpy.ndarray]

    !!! Example
        ```python
        ToNumpy(PipelineStep)

        ## As this is decorated with @SequentialIterator, it can be partially initialised

        partialToNumpy = ToNumpy()
        partialToNumpy(PipelineStep)
        ```
    """
    def __init__(self, index: DataStep):
        """DataOperation to convert data to [np.array][numpy.ndarray]
        
        Args:
            index (DataStep): 
                Underlying DataStep to retrieve Data from
        """        
        super().__init__(index, apply_func=self._convert_xarray_to_numpy, undo_func=self._convert_numpy_to_xarray)
        self._records = []

    def _distill_dataset(self, dataset: xr.Dataset) -> dict:
        """Distill Dataset metadata into a dictionary with which a [np.array][numpy.ndarray] can be rebuilt

        Args:
            dataset (xr.Dataset): 
                Reference Dataset
        
        Returns:
            (dict): 
                Dictionary containing `dims`, `coords`, `attrs` and `shape`
        """        
        dims = list(dataset.coords)
        coords = {}
        attrs = dataset.attrs

        for dim in dims:
            coords[dim] = dataset[dim].values

        variables = list(dataset.data_vars)
        coords["Variables"] = variables
        shape = (len(variables), *dataset[variables[0]].shape)

        return {"dims": dim, "coords": coords, "attrs": attrs, "shape": shape}

    def _set_records(self, datasets: tuple[xr.Dataset] | xr.Dataset) -> None:
        """Set and store records from given datasets
        
        
        Args:
            datasets (tuple[xr.Dataset] | xr.Dataset): 
                Dataset/s to save records from
        
        Raises:
            TypeError: 
                If invalid `datasets` passed
        """        
        if isinstance(datasets, (xr.DataArray, xr.Dataset)):
            if len(self._records) > 0:
                self._records[0] = self._distill_dataset(datasets)
            else:
                self._records.append(self._distill_dataset(datasets))
            return

        elif isinstance(datasets, (tuple)):
            for i, data in enumerate(datasets):
                if len(self._records) > i:
                    self._records[i] = self._distill_dataset(data)
                else:
                    self._records.append(self._distill_dataset(data))
            return
                    
        raise TypeError(f"Unable to get records of {type(datasets)}")

    def _convert_xarray_to_numpy(self, datasets: tuple[xr.Dataset] | xr.Dataset) -> np.ndarray | tuple[np.ndarray]:
        
        """Convert a given dataset/s to [np.array/s][numpy.ndarray]
        
        Args:
            datasets (tuple[xr.Dataset] | xr.Dataset):
                Dataset/s to convert into arrays

        Raises:
            TypeError: 
                If invalid `datasets` passed
                
        Returns:
            (np.ndarray | tuple[np.ndarray]): 
                Generated array/s from Dataset/s
        """        
        self._set_records(datasets)

        ### Convert a given xarray object into an array
        def convert(dataset: xr.DataArray | xr.Dataset) -> np.ndarray:
            if isinstance(dataset, xr.DataArray):
                return dataset.to_numpy()
            if isinstance(dataset, xr.Dataset):
                return np.stack([dataset[var].to_numpy() for var in dataset], axis=1)

        if isinstance(datasets, (xr.DataArray, xr.Dataset)):
            return convert(datasets)

        elif isinstance(datasets, (tuple)):
            return (convert(data) for data in datasets)

        raise TypeError(f"Unable to convert data of {type(datasets)} to np.ndarray")


    def _rebuild_arrays(self, numpy_array: np.ndarray, xarray_distill: dict) -> xr.Dataset:
        """Rebuild a given [np.array][numpy.ndarray] into an [Dataset][xarray.Dataset] using a metadata dictionary
        
        
        Args:
            numpy_array (np.ndarray):
                Numpy array to rebuild
            xarray_distill (dict):
                Dictionary defining `dims`, `coords`, `shape` with which to create the [Dataset][xarray.Dataset]
        
        Returns:
            (xr.Dataset): 
                Rebuilt Dataset
        """        
        data_vars = {}

        coords = dict(xarray_distill["coords"])
        variables = coords.pop("Variables")

        for i in range(numpy_array.shape[xarray_distill["dims"].index("Variables")]):
            data = np.take(
                numpy_array, i, axis=xarray_distill["dims"].index("Variables")
            )
            data_vars[variables[i]] = (coords, data)

        ds = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs=xarray_distill.get("attrs",{}),
        )
        return ds

    def _convert_numpy_to_xarray(self, data: np.ndarray) -> xr.Dataset | tuple[xr.Dataset]:
        
        if isinstance(data, (np.ndarray)):
            return self._rebuild_arrays(data, self._records[0])

        elif isinstance(data, (tuple)):
            return (self._rebuild_arrays(np_data, self._records[i]) for i, np_data in enumerate(data))


    def __iter__(self) -> tuple[np.ndarray]:
        for datasets in self.iterator:
            yield self._convert_xarray_to_numpy(datasets)

    @property
    def __doc__(self):
        return f"Convert to numpy arrays"
