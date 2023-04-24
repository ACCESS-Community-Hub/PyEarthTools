import functools
import time
from itertools import zip_longest
from typing import Union

import numpy as np
import xarray as xr

from dset.training.data.templates import BaseDataOperation, DataInterface
from dset.training.data.sequential import Sequential, SequentialIterator

try:
    from dset.utils.data import Tesselator
except ImportError:
    from ._patching import Tesselator


from dset.data import DataIndex, Collection


@SequentialIterator
class PatchingDataIndex(BaseDataOperation):
    """
    Seperate data into np array patches from a data source
    """
    def __init__(
        self,
        index: DataInterface,
        kernel_size: tuple[int, int] | int,
        stride_size: tuple[int, int] | int = None,
        padding: str = "constant",
    ) -> None:
        """
        Provide functionality to patch data for consumption by an ML Model.

        Return patches will be of shape (Patch, ..., *kernel_size)

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
        super().__init__(index)
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
        kernel_size: tuple[int, int] | int = None,
        stride_size: tuple[int, int] | int = None,
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

    def __apply_tesselators(self, datasets: tuple[xr.Dataset] | xr.Dataset):
        """
        Apply Tesselators on Datasets
        """

        if isinstance(datasets, (xr.Dataset, xr.DataArray)):
            datasets = datasets.compute()
            for patch in self._get_tesselators(1)[0].patch(datasets):
                yield (patch,)

        elif isinstance(datasets, (list, tuple, Collection)):
            tesselators = self._get_tesselators(len(datasets))
            for patches in zip(
                *(tesselators[i].patch(datasets[i]) for i in range(len(datasets)))
            ):
                yield patches
        else:
            raise NotImplementedError(f"Cannot apply tesselation to {type(datasets)!r}")


    def __iter__(self) -> tuple[np.ndarray]:
        for datasets in self.index:
            for i in self.__apply_tesselators(datasets):
                if len(i) == 1:
                    yield i[0]
                yield i

    def __getitem__(self, idx: str):
        datasets = self.index[idx]
        patches = self.__apply_tesselators(datasets)

        result = tuple(map(np.array, zip(*patches)))
        if len(result) == 1:
            return result[0]
        else:
            return result

    def get_before_patching(self, idx: str):
        return self.index[idx]

    def undo(
        self,
        data: np.ndarray | tuple[np.ndarray],
        override_index: int = None,
        **kwargs,
    ) -> xr.Dataset | tuple[xr.Dataset]:
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
            datasets = [tesselators[i].stitch(data[i]) for i in range(len(data))]
        else:
            raise NotImplementedError(f"What is {type(data)}")

        return self.index.undo(datasets, **kwargs)

    def _formatted_name(self):
        desc = f"Kernel_size {self.kernel_size}. Stride_size {self.stride_size or self.kernel_size}"
        return super()._formatted_name(desc)


    def __copy__(self):
        return PatchingDataIndex(
            self.index, self.kernel_size, self.stride_size, self.padding
        )