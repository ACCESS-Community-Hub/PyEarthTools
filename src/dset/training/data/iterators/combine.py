import functools
import time
from itertools import zip_longest
from typing import Union

import numpy as np
import xarray as xr

from dset.training.data.templates import DataIterator
from dset.training.data import DataIterator
from dset.training.data.sequential import Sequential, SequentialIterator


@SequentialIterator
class CombineDataIndex(DataIterator):
    def __init__(
        self, *data_iterators: DataIterator
    ) -> None:
        """
        Combine Multiple DataIterator together, alternating between samples from each

        Parameters
        ----------
        *data_iterators
            DataIterator's to combine
        """

        super().__init__()

        if not isinstance(data_iterators, (list, tuple)):
            data_iterators = [data_iterators]
        self.data_iterators: list[DataIterator]
        self.data_iterators = data_iterators

    @functools.wraps(DataIterator.set_iterable)
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
        return self.data_iterators[iterator_index].undo(data, *args, **kwargs)

    def _formated_name(self):
        desc = f"Combining {self.data_iterators}"
        padding = lambda name, length_: name + "".join([" "] * (length_ - len(name)))
        desc = desc.replace("\n", "").replace("\t", "").strip()
        formatted = f"{padding(self.__class__.__name__, 30)}{desc}"

        for d_iter in self.data_iterators:
            if hasattr(self.index, '_formatted_name'):
                formatted += f"\n{self.index._formatted_name()}"
        return formatted

