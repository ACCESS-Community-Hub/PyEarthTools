import re
from typing import Union

import numpy as np

import dset.data

from dset.training.data.templates import (
    DataIterationOperator,
    DataIterator,
    DataOperation,
    SequentialIterator,
)
from dset.training.data.utils import get_callable, get_class, get_transforms

@SequentialIterator
class TransformOperation(DataOperation):
    """
    Apply dset.data transforms to data
    """

    def __init__(
        self,
        iterator: DataIterator,
        transforms: dict,
        **kwargs
    ) -> None:

        super().__init__(iterator, self._apply_transforms, undo_func=None, **kwargs)
        self.transforms = get_transforms(transforms)
        self.__doc__ = f"Apply Transforms {self.transforms}"

    def _apply_transforms(self, data):
        return self.transforms(data)