from __future__ import annotations

import numpy as np

from edit.training.data.templates import DataStep, DataIterator
from edit.training.data.sequential import SequentialIterator


@SequentialIterator
class CustomLoader(DataStep):
    """
    DataLoader to batch data
    """
    def __init__(self, index: DataStep, batch_size: int):
        super().__init__(index)

        if not isinstance(batch_size, int):
            raise TypeError(f"'batch_size' must be int, not {type(batch_size)}")
        self.batch_size = batch_size

        self._info_ = dict(batch_size = batch_size)

    def __getitem__(self, idx):
        return self.index(idx)

    def __iter__(self):
        iterator = iter(self.index)

        batch = []
        try:
            while True:
                batch.append(iterator.__next__())
                if len(batch) == self.batch_size:
                    yield tuple(map(np.stack, zip(*batch)))
                batch = []

        except StopIteration:
            return tuple(map(np.stack, zip(*batch)))