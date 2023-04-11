import functools
import time
from itertools import zip_longest
from typing import Union

import numpy as np
import xarray as xr

from dset.training.data.templates import DataStep, DataIterator, SequentialIterator


@SequentialIterator
class Iterator(DataIterator):
    """Provide Date Based iteration"""
    def __init__(self, index: DataStep, catch: Union[tuple[Exception], Exception] = None) -> None:
        super().__init__(index, catch)
