import re
from typing import Union

import numpy as np

import edit.data

from edit.training.data.templates import (
    DataIterator,
    DataStep,
    DataOperation,
)
from edit.training.data.utils import get_callable, get_class, get_transforms
from edit.training.data.sequential import Sequential, SequentialIterator


@SequentialIterator
class TransformOperation(DataOperation):
    """
    Apply edit.data transforms to data
    """

    def __init__(self, iterator: DataIterator, transforms: dict, **kwargs) -> None:
        super().__init__(iterator, self._apply_transforms, undo_func=None, **kwargs)
        self.transforms = get_transforms(transforms)
        self.__doc__ = f"Apply Transforms {self.transforms}"

    def _apply_transforms(self, data):
        return self.transforms(data)
