from __future__ import annotations

import xarray as xr

from edit.data import DataIndex, OperatorIndex

from edit.training.data.utils import get_transforms
from edit.training.data.templates import TrainingDataIndex, TrainingOperatorIndex
from edit.training.data.sequential import SequentialIterator


@SequentialIterator
class CombineIndex(TrainingDataIndex):
    """
    DataIndex which combines data into a tuple from other indexes


    !!! Example
        ```python
        CombineIndex(PipelineStep)

        ## As this is decorated with @SequentialIterator, it can be partially initialised

        partialCombineIndex = MergeIndex()
        partialCombineIndex(PipelineStep)
        ```
    """

    def __init__(
        self,
        indexes: list | dict | OperatorIndex | TrainingOperatorIndex | DataIndex | OperatorIndex = {}, **kwargs,
    ):
        """DataIndex which combines data into a tuple from other indexes


        Args:
            indexes (list | dict | OperatorIndex | TrainingOperatorIndex | DataIndex | OperatorIndex):
                Indexes in which to interpolate together and return, can be fully defined or dictionary defined
        """
        indexes.update(kwargs)
        super().__init__(
            indexes,
        )

        self.__doc__ =  "Combine Indexes"
        self._info_ = {idx.__name__: idx.__doc__ for idx in indexes}

    def get(self, querytime, **kwargs) -> tuple[xr.Dataset]:
        """
        Get Data at given time from all given indexes

        Args:
            querytime (Any):
                Time to retrieve data at

        Returns:
            (tuple[xr.Dataset]):
                Tuple of [xr.Dataset][xarray.Dataset] containing data from all indexes interpolated together
        """

        return tuple(index(querytime) for index in self.indexes)

