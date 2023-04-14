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

from dset.training.data.utils import get_callable, get_class

def get_transforms(sources: dict, order : list = None):
    indexes = []
    
    order = order or list(sources.keys())

    for transform in order:
        kwargs = sources[transform]
        data_transform = None

        transform = re.sub(r'\[[0-9]*\]', '', transform)

        try:
            data_transform = get_class(dset.data.transform, transform)
        except:
            pass
        
        if not data_transform:
            for alterations in ["__main__.", "","dset.data.transform", "dset.data."]:
                try:
                    data_transform = get_callable(alterations + transform)
                except (ModuleNotFoundError, ImportError, AttributeError, ValueError):
                    pass
                if data_transform:
                    break
                
        if not data_transform:
            raise ValueError(f"Unable to load {transform!r}")

        if not callable(data_transform):
            if hasattr(data_transform, transform.split(".")[-1]):
                data_transform = getattr(data_transform, transform.split(".")[-1])
            else:
                raise TypeError(
                    f"{transform!r} is a {type(data_transform)}, must be callable"
                )
        try:
            indexes.append(data_transform(**kwargs))
        except Exception as e:
            raise RuntimeError(f"Initialising {transform} raised {e}")
    return dset.data.transform.TransformCollection(indexes)
            

@SequentialIterator
class Transforms(DataOperation):
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