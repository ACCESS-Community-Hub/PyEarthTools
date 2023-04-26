import functools
import time
from itertools import zip_longest
from typing import Union

import numpy as np
import xarray as xr

from edit.training.data.templates import DataIterator
from edit.training.data.iterators.temporal import DataInterface
from edit.training.data.sequential import Sequential, SequentialIterator


@SequentialIterator
class ToNumpy(DataIterator):
    def __init__(self, data_interface: DataInterface) -> None:
        super().__init__(data_interface)
        self._records = []

    def _distill_dataset(self, dataset: xr.Dataset):
        dims = list(dataset.coords)
        coords = {}
        attrs = dataset.attrs

        for dim in dims:
            coords[dim] = dataset[dim].values

        variables = list(dataset.data_vars)
        coords["Variables"] = variables
        shape = (len(variables), *dataset[variables[0]].shape)

        return {"dims": dim, "coords": coords, "attrs": attrs, "shape": shape}

    def _set_records(self, datasets):
        if isinstance(datasets, (xr.DataArray, xr.Dataset)):
            if len(self._records) > 0:
                self._records[0] = self._distill_dataset(datasets)
            else:
                self._records.append(self._distill_dataset(datasets))
        elif isinstance(datasets, (tuple)):
            for i, data in enumerate(datasets):
                if len(self._records) > i:
                    self._records[i] = self._distill_dataset(data)
                else:
                    self._records.append(self._distill_dataset(data))

    def _convert_xarray_to_numpy(self, datasets):
        def convert(dataset):
            if isinstance(dataset, xr.DataArray):
                return dataset.to_numpy()
            if isinstance(dataset, xr.Dataset):
                return np.stack([dataset[var].to_numpy() for var in dataset], axis=1)

        if isinstance(datasets, (xr.DataArray, xr.Dataset)):
            return convert(datasets)
        if isinstance(datasets, (tuple)):
            for i, data in enumerate(datasets):
                datasets.append(convert(data))
            return datasets

    def _convert_numpy_to_xarray(self, numpy_array: np.array, xarray_distill: dict):
        variables = xarray_distill["coords"]["Variables"]
        data_vars = {}

        coords = dict(xarray_distill["coords"])
        coords.pop("Variables")

        # if 'time' in coords: coords['time'] = np.atleast_1d(coords['time'])
        for i in range(numpy_array.shape[xarray_distill["dims"].index("Variables")]):
            data = np.take(
                numpy_array, i, axis=xarray_distill["dims"].index("Variables")
            )
            data_vars[variables[i]] = (coords, data)

        ds = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs=xarray_distill["attrs"],
        )
        return ds

    def _rebuild_records(self, data: np.array):
        if isinstance(data, (np.array)):
            return self._convert_numpy_to_xarray(data, self._records[0])

        elif isinstance(data, (tuple)):
            datasets = []
            for i, np_data in enumerate(data):
                datasets.append(
                    self._convert_numpy_to_xarray(np_data, self._records[i])
                )
            return datasets

    def __getattr__(self, key):
        return getattr(self.iterator, key)

    def __iter__(self) -> tuple[np.ndarray]:
        for datasets in self.iterator:
            self._set_records(datasets)
            yield self._convert_xarray_to_numpy(datasets)

    def __getitem__(self, idx: str):
        datasets = self.iterator[idx]
        return self._convert_xarray_to_numpy(datasets)

    def undo(
        self, data: np.ndarray | tuple[np.ndarray]
    ) -> xr.Dataset | tuple[xr.Dataset]:
        return self.iterator.undo(self._convert_numpy_to_xarray(data))

    def _formatted_name(self):
        desc = f"Convert to numpy arrays"
        return super()._formatted_name(desc)
