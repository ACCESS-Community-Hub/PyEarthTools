import functools
import time
from itertools import zip_longest
from typing import Union

import numpy as np
import xarray as xr

from dset.training.data.templates import DataIterator, SequentialIterator
from dset.training.data.indexes.data_interface import DataInterface, Data_Interface
from dset.training.data.indexes.patching import Tesselator
from dset.training.trainer.trainer import DSETTrainerWrapper


class PatchingUpdate:
    """
    Patching Update Context Manager.

    So that any changes to patching can be reversed.
    """

    def __init__(
        self,
        iterator: Union["PatchingDataIndex", DSETTrainerWrapper],
        kernel_size: Union[tuple[int, int], int] = None,
        stride_size: Union[tuple[int, int], int] = None,
    ):
        """
        Update Patching Config

        Parameters
        ----------
        iterator
            Iterator in which to update
        kernel_size, optional
            New kernel_size, by default None
        stride_size, optional
            New Stride size, by default None

        Raises
        ------
        RuntimeError
            If iterator is not a PatchingDataIndex
        """
        if isinstance(iterator, DSETTrainerWrapper):
            iterator = iterator.valid_iterator or iterator.train_iterator

        if not hasattr(iterator, "get_patching"):
            raise RuntimeError("DataIterator does not seem to be a PatchingDataIndex.")

        self.iterator = iterator

        self._patching_initial = iterator.get_patching()
        self._new_patching = (kernel_size, stride_size)

    def __enter__(self):
        self.iterator.update_patching(*self._new_patching)

    def __exit__(self, *args):
        self.iterator.update_patching(*self._patching_initial)


@SequentialIterator
class PatchingDataIndex(DataIterator):
    def __init__(
        self,
        data_interface: DataInterface,
        kernel_size: Union[tuple[int, int], int],
        stride_size: Union[tuple[int, int], int] = None,
        padding: str = "constant",
    ) -> None:
        """
        Provide functionality to patch data for consumption by an ML Model.

        Return patches will be of shape (Patch, Channels, Time, *kernel_size)

        Parameters
        ----------
        data_interface
            DataInterface which interfaces with the DataIndex/s
        kernel_size
            Kernel size of the data to be returned
        stride_size, optional
            Stride size of the data, by default None
        padding, optional
            Padding method to use. Must be of np.pad, by default 'constant'
        """
        super().__init__()
        self.data_interface = data_interface
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.padding = padding

        self._tesselators = []

    def _get_tesselators(self, number: int) -> tuple[Tesselator]:
        """
        Retrieve a set number of tesselators, creating new ones if needed
        """
        return_values = []
        for i in range(number):
            if i < len(self._tesselators):
                return_values.append(self._tesselators[i])
            else:
                self._tesselators.append(
                    Tesselator(self.kernel_size, self.stride_size, self.padding)
                )
                return_values.append(self._tesselators[-1])

        return return_values

    def get_patching(self):
        """
        Get Patching Setup

        Returns
        -------
            Patching Info - tuple[kernel_size, stride_size]
        """
        return self.kernel_size, self.stride_size

    def update_patching(
        self,
        kernel_size: Union[tuple[int, int], int] = None,
        stride_size: Union[tuple[int, int], int] = None,
    ):
        """
        Reset Tesselators and update patching configs.

        Parameters
        ----------
        kernel_size, optional
            New kernel_size, by default None
        stride_size, optional
            New stride size, by default None
        """
        self._tesselators = []
        self.kernel_size = kernel_size or self.kernel_size
        self.stride_size = stride_size

    def __apply_tesselators(self, datasets: Union[tuple[xr.Dataset], xr.Dataset]):
        """
        Apply Tesselators on Datasets
        """
        if isinstance(datasets, (xr.Dataset, xr.DataArray)):
            for patch in self._get_tesselators(1)[0].patch(datasets):
                yield (patch,)

        elif isinstance(datasets, (list, tuple)):
            tesselators = self._get_tesselators(len(datasets))
            for patches in zip(
                *(tesselators[i].patch(datasets[i]) for i in range(len(datasets)))
            ):
                yield patches
        else:
            raise NotImplementedError(f"What is {type(datasets)}")

    def __getattr__(self, key):
        return getattr(self.data_interface, key)

    def __iter__(self) -> tuple[np.ndarray]:
        for datasets in self.data_interface:
            for i in self.__apply_tesselators(datasets):
                if len(i) == 1:
                    yield i[0]
                yield i

    def __getitem__(self, idx: str):
        datasets = self.data_interface[idx]
        patches = self.__apply_tesselators(datasets)

        result = tuple(map(np.array, zip(*patches)))
        if len(result) == 1:
            return result[0]
        else:
            return result

    def undo(
        self,
        data: Union[np.ndarray, tuple[np.ndarray]],
        override_index: int = None,
        data_index: int = None,
    ) -> Union[xr.Dataset, tuple[xr.Dataset]]:
        """
        Undo patching done to Datasets. Automatically stitching them together.

        Can be run on direct output of self[]

        Parameters
        ----------
        data
            Arrays to stitch back together
        override_index, optional
            Override of which tesselator to use, by default None
        data_index, optional
            Override to pass along to DataInterface on which Unnormalise to use,
                by default None

        Returns
        -------
            Data stitched together

        Raises
        ------
        NotImplementedError
            If data is not recognised
        """
        if isinstance(data, np.ndarray):
            if override_index == -1:
                override_index = len(self._tesselators)
            datasets = self._get_tesselators(
                override_index + 1 if override_index else 1
            )[override_index or 0].stitch(data)

        elif isinstance(data, (list, tuple)):
            tesselators = self._get_tesselators(len(data))
            datasets = (tesselators[i].stitch(data[i]) for i in range(len(data)))
        else:
            raise NotImplementedError(f"What is {type(data)}")

        return self.data_interface.undo(datasets, override_index=data_index)

    def _formatted_name(self):
        padding = lambda name, length_: name + "".join([" "] * (length_ - len(name)))
        desc = f"Kernel_size {self.kernel_size}. Stride_size {self.stride_size or self.kernel_size}"
        desc = desc.replace("\n", "").replace("\t", "").strip()
        return f"{padding(self.__class__.__name__, 30)}{desc}\n{self.data_interface._formatted_name()}"

    def __repr__(self):
        string = "DataIterator with the following Operations:"
        operations = self._formatted_name()
        operations = "\n".join(["\t* " + oper for oper in operations.split("\n")])
        return f"{string}\n{operations}"

    def __copy__(self):
        return PatchingDataIndex(
            self.data_interface, self.kernel_size, self.stride_size, self.padding
        )


@SequentialIterator
class AsNumpy(DataIterator):
    def __init__(self, data_interface: DataInterface) -> None:
        self.data_interface = data_interface

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
        return getattr(self.data_interface, key)

    def __iter__(self) -> tuple[np.ndarray]:
        for datasets in self.data_interface:
            self._set_records(datasets)
            yield self._convert_xarray_to_numpy(datasets)

    def __getitem__(self, idx: str):
        datasets = self.data_interface[idx]
        return self._convert_xarray_to_numpy(datasets)

    def undo(
        self, data: Union[np.ndarray, tuple[np.ndarray]]
    ) -> Union[xr.Dataset, tuple[xr.Dataset]]:
        return self.data_interface.undo(self._convert_numpy_to_xarray(data))

    def _formated_name(self):
        padding = lambda name, length_: name + "".join([" "] * (length_ - len(name)))
        desc = f"Convert to numpy arrays"
        desc = desc.replace("\n", "").replace("\t", "").strip()
        return f"{padding(self.__class__.__name__, 20)}{desc}\n{self.data_interface._formatted_name()}"

    def __repr__(self):
        string = "DataIterator with the following Operations:"
        operations = self._formatted_name()
        operations = "\n".join(["\t* " + oper for oper in operations.split("\n")])
        return f"{string}\n{operations}"


@SequentialIterator
class CombineDataIndex(DataIterator):
    def __init__(
        self, *data_iterators: Union[DataIterator, tuple[DataIterator]]
    ) -> None:
        """
        Combine Multiple DataIterators together, alternating between samples from each

        Parameters
        ----------
        *data_iterators
            DataIterators to combine

        """

        super().__init__()

        if not isinstance(data_iterators, (list, tuple)):
            data_iterators = [data_iterators]
        self.data_iterators: list[DataIterator]
        self.data_iterators = data_iterators

    @functools.wraps(Data_Interface.set_iterable)
    def set_iterable(self, *args, **kwargs):
        for iterator in self.data_iterators:
            iterator.set_iterable(*args, **kwargs)

    def __iter__(self):
        for data_collections in zip_longest(*self.data_iterators, fillvalue=None):
            data_collections = list(data_collections)
            while None in data_collections:
                data_collections.remove(None)

            for data in data_collections:
                yield data

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.data_iterators[idx]
        elif isinstance(idx, tuple):
            next_idx = idx[1:]
            if len(next_idx) == 1:
                next_idx = next_idx[0]
            return self[idx[0]].__getitem__(next_idx)
        raise ValueError

    def undo(self, data, iterator_index: int, *args, **kwargs):
        return self.data_iterators[iterator_index].undo(data,*args, **kwargs)

    def _formated_name(self):
        padding = lambda name, length_: name + "".join([" "] * (length_ - len(name)))
        desc = f"Combining {self.data_iterators}"
        desc = desc.replace("\n", "").replace("\t", "").strip()

        string = f"{padding(self.__class__.__name__, 20)}"
        for d_iter in self.data_iterators:
            string += f"\n\t{d_iter._formatted_name()}"
        return string

    def __repr__(self):
        string = "DataIterator with the following Operations:"
        operations = self._formatted_name()
        operations = "\n".join(["\t* " + oper for oper in operations.split("\n")])
        return f"{string}\n{operations}"
