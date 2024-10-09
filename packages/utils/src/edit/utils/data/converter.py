# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

# type: ignore[reportPrivateImportUsage]


from __future__ import annotations
from abc import ABCMeta, abstractmethod
import importlib.util
import json
from pathlib import Path
from typing import Any, Union, Literal

from collections import OrderedDict
import warnings

import importlib


import numpy as np
import xarray as xr

XR_OBJECT = Union[xr.DataArray, xr.Dataset]

DISTILL_KEYS = Literal[
    "dims",
    "coords",
    "attrs",
    "var_attrs",
    "shape",
    "encoding",
]


class xarrayConverter(metaclass=ABCMeta):
    """
    Stateful xarray converter.

    This class will record records of the attributes and coordinates of the
    incoming datasets, which are then used to rebuild the array.

    This operates on a first in first out (FIFO) approach with storing, and
    removing records. So if two datasets are converter, it is expected that they
    will come be converted back to xarray in the same order.

    Cannot be used directly, use either `NumpyConverter` or `DaskConverter`

    Examples:
        >>> converter = NumpyConverter()
        >>> np_data_1 = converter(dataset_1)
        >>> np_data_2 = converter(dataset_2)

        >>> converter(np_data_1) == dataset_1
        True
        >>> converter(np_data_2) == dataset_2
        True

        >>> np_data_2 = converter(dataset_2)
        >>> np_data_1 = converter(dataset_1)

        >>> converter(np_data_1) == dataset_1
        False
        >>> converter(np_data_2) == dataset_2
        False
    """

    def __init__(self, warn: bool = True):
        """
        Converter to and from xarray objects

        Args:
            warn (bool, optional):
                Warn on incorrect shape. Defaults to True.
        """
        self._records = []
        self._warn = warn

    @property
    def records(self) -> list[dict[DISTILL_KEYS, Any]]:
        return self._records

    @records.setter
    def records(self, records: list[dict[DISTILL_KEYS, Any]]):
        self._records = records

    def save_records(self, filepath: str | Path):
        def parse(value):
            if isinstance(value, np.ndarray):
                if value.dtype.__class__.__name__ == "DateTime64DType":
                    return list(map(str, value))
                return list(map(float, value.tolist()))

            if isinstance(value, np.integer):
                return int(value)

            try:
                if np.isnan(value).all():
                    return None
            except TypeError:
                pass
            if isinstance(value, np.floating):
                return float(value)

            if isinstance(value, np.dtype):
                return str(value)
            if isinstance(value, (list, tuple)):
                return type(value)(map(parse, value))
            if isinstance(value, dict):
                return {key: parse(val) for key, val in value.items()}
            return value

        _records = parse(list(self.records))
        json.dump(_records, open(filepath, "w"), indent=3)

    def load_records(self, filepath: str | Path):
        if Path(filepath).exists():
            self.records = json.load(open(filepath))
            return True
        return False

    def _distill_dataset(self, dataset: XR_OBJECT) -> dict[DISTILL_KEYS, Any]:
        """Distill Dataset metadata into a dictionary with which a [np.array][numpy.ndarray] can be rebuilt

        Args:
            dataset (XR_OBJECT):
                Reference Dataset

        Returns:
            (dict):
                Dictionary containing `dims`, `coords`, `attrs` and `shape`
        """
        if isinstance(dataset, xr.DataArray):
            dataset = dataset.to_dataset(name="Data")
        coords = OrderedDict()
        attrs = dataset.attrs

        variables = list(dataset.data_vars)

        shape = (len(variables), *dataset[variables[0]].shape)

        dims = [None] * (len(dataset.coords) + 1)

        use_shape = list(shape)
        for coord in dataset.coords:
            size = len(dataset[coord])
            try:
                dims[use_shape.index(size)] = coord
            except ValueError as e:
                raise RuntimeError(
                    "Cannot record coordinate, currently converter can only handle datasets with variables of the same dimensions."
                ) from e
            use_shape[use_shape.index(size)] = 1e10

        while None in dims:
            dims.remove(None)

        for dim in dims:
            coords[dim] = dataset[dim].values
        coords["Variables"] = variables
        var_attrs = {var: dataset[var].attrs for var in [*variables, *dims]}

        ## Record Relevant encoding information
        encoding = {}
        relevant_encoding = ["units", "dtype", "calendar", "_FillValue", "scale_factor", "add_offset", "missing_value"]

        for var in (*dims, *variables):
            encoding[var] = {}
            for rel in set(relevant_encoding).intersection(set(list(dataset[var].encoding.keys()))):
                encoding[var][rel] = dataset[var].encoding[rel]
        dims = list(("Variables", *dims))

        return {
            "dims": dims,
            "coords": coords,
            "attrs": attrs,
            "var_attrs": var_attrs,
            "shape": shape,
            "encoding": encoding,
        }

    def _set_records(
        self, datasets: tuple[XR_OBJECT, ...] | list[XR_OBJECT] | XR_OBJECT, replace: bool = False
    ) -> None:
        """Set and store records from given datasets

        This will append to the `_records`

        Args:
            datasets (tuple[XR_OBJECT] | list[XR_OBJECT] | XR_OBJECT):
                Dataset/s to save records from
            replace (bool, optional):
                Whether to reset the records each time. Defaults to False.

        Raises:
            TypeError:
                If invalid `datasets` passed
        """

        if replace:
            self.records = []

        if isinstance(datasets, (tuple, list)):
            list(map(self._set_records, datasets))
            return

        elif isinstance(datasets, (xr.DataArray, xr.Dataset)):
            self._records.append(self._distill_dataset(datasets))
            return

        raise TypeError(f"Unable to get records of {type(datasets)}")

    @abstractmethod
    def convert_from_xarray(
        self,
        data: tuple[XR_OBJECT, ...] | list[XR_OBJECT] | XR_OBJECT,
        replace: bool = False,
    ): ...

    @abstractmethod
    def convert_to_xarray(
        self, data: np.ndarray | tuple[np.ndarray, ...], pop: bool = True
    ) -> xr.Dataset | tuple[xr.Dataset]: ...

    def __call__(self, data: xr.Dataset | np.ndarray | tuple) -> xr.Dataset | np.ndarray | tuple:
        """
        Convert either a numpy.array or xarray dataset to the other form.


        Args:
            data (xr.Dataset | np.ndarray | tuple):
                Data to convert

        Returns:
            (xr.Dataset | np.ndarray | tuple):
                Converted data
        """
        if isinstance(data, (xr.Dataset, xr.DataArray)):
            return self.convert_from_xarray(data)
        return self.convert_to_xarray(data)


class NumpyConverter(xarrayConverter):

    def convert_from_xarray(
        self,
        data: tuple[XR_OBJECT, ...] | list[XR_OBJECT] | XR_OBJECT,
        replace: bool = False,
    ) -> np.ndarray | tuple[np.ndarray]:
        """Convert a given dataset/s to [np.array/s][numpy.ndarray]

        Reminder, this class operates with a FIFO approach, each incoming
        dataset's records will be saved, and popped out when being rebuilt.
        Unless `replace` is True, then will replace instead.

        Args:
            data (tuple[xr.Dataset] | xr.Dataset):
                data/s to convert into arrays
            replace (bool, optional):
                Whether to replace entries, instead of inserting them. Defaults to False

        Raises:
            TypeError:
                If invalid `data` passed

        Returns:
            (np.ndarray | tuple[np.ndarray]):
                Generated array/s from Dataset/s
        """
        self._set_records(data, replace=replace)

        ### Convert a given xarray object into an array
        def convert(dataset: xr.DataArray | xr.Dataset) -> np.ndarray:
            if isinstance(dataset, xr.DataArray):
                return dataset.to_numpy()
            if isinstance(dataset, xr.Dataset):
                return np.stack([dataset[var].to_numpy() for var in dataset], axis=0)

        if isinstance(data, (xr.DataArray, xr.Dataset)):
            data = data.compute()
            return convert(data)

        elif isinstance(data, (tuple)):
            return tuple(map(convert, data))  # type: ignore

        raise TypeError(f"Unable to convert data of {type(data)} to np.ndarray")

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

        full_coords = dict(coords)

        if not numpy_array.shape == tuple(xarray_distill["shape"]):
            if self._warn:
                warnings.warn(
                    f"Incoming and expected shape don't match. {numpy_array.shape} != {xarray_distill['shape']}. Dropping conflicting coordinates.",
                    RuntimeWarning,
                )
            for index in range(len(numpy_array.shape)):
                if not numpy_array.shape[index] == xarray_distill["shape"][index]:
                    full_coords.pop(list(coords.keys())[index - 1])

        variables = coords.pop("Variables")
        full_coords.pop("Variables", None)

        ar = np
        try:
            import dask.array as da
        except (ImportError, ModuleNotFoundError):
            ar = da

        for i in range(numpy_array.shape[xarray_distill["dims"].index("Variables")]):
            data = ar.take(numpy_array, i, axis=xarray_distill["dims"].index("Variables"))
            data_vars[variables[i]] = (
                coords,
                data,
                xarray_distill["var_attrs"][variables[i]],
            )

        try:
            ds = xr.Dataset(
                data_vars=data_vars,
                coords=full_coords,
                attrs=xarray_distill["attrs"],
            )
            ds.attrs.update(xarray_distill["attrs"])

        except ValueError as e:
            raise ValueError(
                f"An error occurred converting data back to a NumPy array. Incoming shape is {numpy_array.shape}"
            ) from e

        for var, encod in xarray_distill["encoding"].items():
            if var in ds:
                ds[var].encoding.update(encod)
        for var, attrs in xarray_distill["var_attrs"].items():
            if var in ds:
                ds[var].attrs.update(attrs)
        return ds

    def convert_to_xarray(
        self, data: np.ndarray | tuple[np.ndarray, ...], pop: bool = True
    ) -> xr.Dataset | tuple[xr.Dataset]:
        """
        Convert [array/s][numpy.ndarray] into [Dataset/s][xarray.Dataset] inferring metadata from saved records.

        Reminder, this class operates on a FIFO approach, records will be popped from the saved records, unless turned off.

        !!! Warning
            If a tuple of datasets was passed to [convert_xarray_to_numpy][edit.pipeline.operations.to_numpy.NumpyConverter.convert_xarray_to_numpy]
            and they are different, it is best to pass a tuple to this function replicating the order

        Args:
            data (np.ndarray):
                [array/s][numpy.ndarray] to convert back to [Dataset/s][xarray.Dataset]
            pop (bool, optional):
                Whether to pop records from _records. Defaults to True

        Returns:
            (xr.Dataset | tuple[xr.Dataset]):
                Rebuilt [Dataset/s][xarray.Dataset]
        """
        if not self._records:
            raise RuntimeError("Data hasn't been converted to arrays with this. So data cannot be undone")

        if isinstance(data, (tuple, list)):
            if not pop:
                return tuple(self._rebuild_arrays(np_data, self._records[i]) for i, np_data in enumerate(data))
            return tuple(map(self.convert_to_xarray, data))

        if len(data) == 0:
            raise RuntimeError("Xarray records is empty, cannot rebuild any more data")
        if pop:
            record = self._records.pop(0)
        else:
            record = self._records[0]

        return self._rebuild_arrays(data, record)

        raise TypeError(f"Cannot convert {type(data)} to xarray")


class DaskConverter(NumpyConverter):
    def __new__(cls, *args, **kwargs):
        try:
            importlib.util.find_spec("dask")
            return super().__new__(cls)
        except ValueError:
            pass
        raise ImportError("Cannot use `DaskConverter` as dask could not be imported.")

    def convert_from_xarray(
        self,
        data: tuple[XR_OBJECT, ...] | list[XR_OBJECT] | XR_OBJECT,
        replace: bool = False,
    ) -> "dask.array.Array | tuple[dask.array.Array, ...]":
        """Convert a given dataset/s to dask arrays

        Reminder, this class operates with a FIFO approach, each incoming
        dataset's records will be saved, and popped out when being rebuilt.
        Unless `replace` is True, then will replace instead.

        Args:
            data (tuple[xr.Dataset] | xr.Dataset):
                data/s to convert into arrays
            replace (bool, optional):
                Whether to replace entries, instead of inserting them. Defaults to False

        Raises:
            TypeError:
                If invalid `data` passed

        Returns:
            (dask.array.Array | tuple[dask.array.Array, ...]):
                Generated array/s from Dataset/s
        """
        import dask.array as da

        self._set_records(data, replace=replace)

        ### Convert a given xarray object into an array
        def convert(dataset: xr.DataArray | xr.Dataset) -> da.Array:
            if isinstance(dataset, xr.DataArray):
                return dataset.data
            if isinstance(dataset, xr.Dataset):
                return da.stack([dataset[var].data for var in dataset], axis=0)

        if isinstance(data, (xr.DataArray, xr.Dataset)):
            return convert(data)

        elif isinstance(data, (tuple)):
            return tuple(map(convert, data))  # type: ignore

        raise TypeError(f"Unable to convert data of {type(data)} to `da.array`")

    def convert_to_xarray(self, data, pop: bool = True):
        import dask.array as da

        # if isinstance(data, da.Array):
        #     data = data.compute()
        return super().convert_to_xarray(data, pop=pop)
