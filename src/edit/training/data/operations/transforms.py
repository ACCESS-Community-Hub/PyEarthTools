from __future__ import annotations

from edit.training.data.templates import (
    DataStep,
    DataOperation,
)

import xarray as xr

from edit.data import Transform
from edit.training.data.utils import get_callable, get_class, get_transforms
from edit.training.data.sequential import Sequential, SequentialIterator


@SequentialIterator
class TransformOperation(DataOperation):
    """
    DataOperation to apply [Transforms][edit.data.Transform] to incoming data

    !!! Example
        ```python
        TransformOperation(PipelineStep, transforms = {'region.lookup':'Australia'})

        ## As this is decorated with @SequentialIterator, it can be partially initialised

        partialTransformOperation = TransformOperation(transforms = {'region.lookup':'Australia'})
        partialTransformOperation(PipelineStep)
        ```
    """
    def __init__(self, iterator: DataStep, transforms: Transform | dict, **kwargs) -> None:
        """
        DataOperation to apply Transforms

        Args:
            iterator (DataStep): 
                Underlying DataStep to retrieve Data from.
            transforms (Transform | dict): 
                Transforms to apply, either fully defined or dictionary defining the transform.
        """        
        super().__init__(iterator, self._apply_transforms, undo_func=None, split_tuples=True, recognised_types=[xr.Dataset, xr.DataArray], **kwargs)
        self.transforms = get_transforms(transforms)
        self.__doc__ = f"Apply Transforms {self.transforms}"
        self._info_ = dict(transforms = transforms)

    def _apply_transforms(self, data):
        return self.transforms(data)
