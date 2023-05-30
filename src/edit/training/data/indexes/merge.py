from __future__ import annotations

from typing import Any, Callable, Union
import warnings
import xarray as xr

from edit.data import transform, TransformCollection, EDITDatetime, operations
from edit.data import OperatorIndex

from edit.training.data.utils import get_transforms
from edit.training.data.templates import TrainingOperatorIndex
from edit.training.data.sequential import SequentialIterator


@SequentialIterator
class MergeIndex(TrainingOperatorIndex):
    """
    OperatorIndex which merges and returns data from any given indexes on the same spatial grid


    !!! Example
        ```python
        MergeIndex(PipelineStep)

        ## As this is decorated with @SequentialIterator, it can be partially initialised

        partialMergeIndex = MergeIndex()
        partialMergeIndex(PipelineStep)
        ```
    """

    def __init__(
        self,
        indexes: list | dict | OperatorIndex,
        data_resolution: tuple[int, tuple[int]] = None,
        transforms: list | dict = TransformCollection(),
    ):
        """OperatorIndex which interpolates any given indexes onto the same spatial grid

        Will retrieve samples with `data_resolution` resolution.

        Args:
            indexes (list | dict | OperatorIndex):
                Indexes in which to interpolate together and return, can be fully defined or dictionary defined
            data_resolution (tuple[int, tuple[int]], optional):
                Sample Interval to pass up, must be of pandas.to_timestep form.
                E.g. (10,'H') - 10 Hours. Defaults to None.
            transforms (list | dict, optional):
                 Other Transforms to apply. Defaults to TransformCollection().
        """

        if isinstance(transforms, dict):
            transforms = get_transforms(transforms)

        base_transforms = TransformCollection(transforms)

        super().__init__(
            indexes,
            base_transforms=base_transforms,
            data_resolution=data_resolution,
            allow_multiple_index=True,
        )

        self.__doc__ =  "Merge Multiple DataIndexes together"

    def get(self, query_time, **kwargs) -> xr.Dataset:
        """
        Get Data at given time from all given indexes, and merge as defined.

        Args:
            query_time (Any):
                Time to retrieve data at

        Returns:
            (xr.Dataset):
                [xr.Dataset][xarray.Dataset] containing data from all indexes merged together
        """

        return xr.merge([index(query_time, transforms=self.base_transforms, **kwargs) for index in self.index])

