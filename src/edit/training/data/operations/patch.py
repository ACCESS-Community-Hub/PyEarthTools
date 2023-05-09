from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from edit.training.data.templates import DataOperation, DataStep
from edit.training.data.sequential import  SequentialIterator

from edit.utils.data import Tesselator
from edit.data import Collection


@SequentialIterator
class PatchingDataIndex(DataOperation):
    """
    DataOperation to patch data into smaller chunks of [np.array][numpy.ndarray].

    Uses the [Tesselator][edit.utils.data.Tesselator] to patch and stitch the data.

    !!! Warning
        Issues may arise with rebuilding the time dimensions of a [Dataset][xarray.Dataset],
        as the original dataset metadata is used.
        
        It is suggested that you find a way to reset the time dimension afterwards.

        However, if using this with [TemporalIterator][edit.training.data.iterators.TemporalIterator],
        a `rebuild_time` function is provided.

    !!! Example
        ```python
        PatchingDataIndex(PipelineStep)

        ## As this is decorated with @SequentialIterator, it can be partially initialised

        partialPatchingDataIndex = PatchingDataIndex()
        partialPatchingDataIndex(PipelineStep)
        ```
    """
    def __init__(
        self,
        index: DataStep,
        kernel_size: tuple[int, int] | int,
        stride_size: tuple[int, int] | int = None,
        padding: str = "constant",
    ) -> None:
        """Patching DataOperation to split data into smaller chunks
        
        Returned patches will be of shape `(Patch, ..., *kernel_size)`

        Args:
            index (DataStep): 
                Underlying DataStep to retrieve Data from
            kernel_size (tuple[int] | int): 
                Kernel size of the data to be returned
            stride_size (tuple[int] | int, optional): 
                Stride size of the data, if not given default to `kernel_size`. Defaults to None.
            padding (str, optional): 
                Padding method to use. Must be of [np.pad][numpy.pad]. Defaults to "constant".
        """        

        super().__init__(
            index, apply_func=self.__apply_func, undo_func=self._undo_tesselators
        )
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.padding = padding

        self._tesselators = []

        self._info_ = dict(kernel_size = kernel_size, stride_size = stride_size, padding = padding)

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

    def get_patching(self) -> tuple[tuple[int] | int]:
        """Get Patching Setup
        
        Returns:
            (tuple[tuple[int] | int]): 
                Patching Info - [kernel_size, stride_size]
        """        
        return self.kernel_size, self.stride_size

    @property
    def patching_config(self):
        class NameSpace:
            def __init__(self, dict) -> None:
                self.dict = dict
                for key, value in dict.items():
                    setattr(self, key, value)
            def __repr__(self) -> str:
                return str(self.dict)
        return Collection(*(NameSpace(tess._coords) for tess in self._tesselators))

    def update_patching(
        self,
        kernel_size: tuple[int, int] | int = None,
        stride_size: tuple[int, int] | int = None,
    ):
        """Reset Tesselators and update patching configs.        
        
        Args:
            kernel_size (tuple[int, int] | int, optional): 
                New kernel_size. Defaults to None.
            stride_size (tuple[int, int] | int, optional): 
                New stride size. Defaults to None.
        """        
        self._tesselators = []
        self.kernel_size = kernel_size or self.kernel_size
        self.stride_size = stride_size

    def _apply_tesselators(self, datasets: tuple[xr.Dataset] | xr.Dataset):
        """
        Apply Tesselators on Datasets
        """

        if isinstance(datasets, (xr.Dataset, xr.DataArray)):
            datasets = datasets.compute()
            for patch in self._get_tesselators(1)[0].patch(datasets):
                yield (patch,)
        elif isinstance(datasets, (np.ndarray)):
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


    def __apply_func(self, data):
        patches = self._apply_tesselators(data)

        result = tuple(map(np.array, zip(*patches)))
        if len(result) == 1:
            return result[0]
        else:
            return result

    def _undo_tesselators(
        self,
        data: np.ndarray | tuple[np.ndarray],
        override_index: int = None,
    ) -> xr.Dataset | tuple[xr.Dataset | np.ndarray] | np.ndarray:
        """Undo patching done to Datasets. Automatically stitching them together.

        !!! Warning
            If a tuple of datasets was passed to [_apply_tesselators][edit.training.data.operations.patch._apply_tesselators]
            and they are different, it is best to pass a tuple to this function replicating the order

        Can be run on direct output of self[]        
        
        Args:
            data (np.ndarray | tuple[np.ndarray]): 
                Arrays to stitch back together
            override_index (int, optional): 
                Override of which tesselator to use. Defaults to None.
        
        Raises:
            TypeError: 
                If data type is not recognised
        
        Returns:
            (xr.Dataset | tuple[xr.Dataset | np.ndarray] | np.ndarray): 
                Data stitched back together
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
            raise TypeError(f"What is {type(data)}")

        return datasets

    def get_before_patching(self, idx: str) -> Any:
        """
        Get Data before patching step from given index        
        
        Args:
            idx (str): 
                Index to retrieve data at
        
        Returns:
            (Any): 
                Data before patching is applied
        """
        return self.index[idx]

    def __iter__(self) -> tuple[np.ndarray]:
        for dataset in self.index:
            for i in self._apply_tesselators(dataset):
                if len(i) == 1:
                    yield i[0]
                else:
                    yield i

    @property
    def __doc__(self):
        return f"Kernel_size {self.kernel_size}. Stride_size {self.stride_size or self.kernel_size}"

    def __copy__(self):
        return PatchingDataIndex(
            self.index, self.kernel_size, self.stride_size, self.padding
        )
