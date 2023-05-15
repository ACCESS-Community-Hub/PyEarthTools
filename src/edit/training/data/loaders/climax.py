from typing import Union
import warnings
import torch
import numpy as np

from edit.training.data.templates import DataStep, DataIterator, DataOperation
from edit.training.data.sequential import SequentialIterator
from edit.training.data.loaders.pytorch import PytorchIterable
from torch.utils.data import IterableDataset


# TODO Remove this and implement specifically for Solar not in EDIT
@SequentialIterator
class ClimaXDataLoader(DataOperation, IterableDataset):
    """
    Connect Data Pipeline with PyTorch IterableDataset for use with ClimaX
    """

    def __init__(self, index: DataStep | DataIterator) -> None:
        super().__init__(index=index, apply_func=self._add_time_dim, undo_func=self._undo_func)
        warnings.warn(f"ClimaX Dataloader will be moved out shortly", DeprecationWarning)
        self._size = 2

    def _undo_func(self, data):
        return data[:self._size]

    def _find_time(self, size = 1):
        if not hasattr(self.index, "sample_interval"):
            raise RuntimeError(f"Not using a known TemporalIterator")

        # if isinstance(data, tuple)
        if size == 1:
            return torch.Tensor([1])
        return torch.Tensor(np.linspace(0,1,size))

    def _add_time_dim(self, data):
        self._size = len(data)
        # extend = data[0].shape[0]
        # return (*data, self._find_time().expand(extend))

        if isinstance(data, tuple):
            if len(data[0].shape) > 4:
                time = self._find_time(size = data[1].shape[-3])
                return (*data, time)
            return (*data, self._find_time())
        else:
            raise NotImplementedError()

    def __iter__(self):
        for data in self.index:
            if isinstance(data, tuple):
                if len(data[0].shape) == 4:
                    time = self._find_time(size = data[1].shape[1])
                    for i in range(data[1].shape[1]):
                        yield (data[0][:,0], data[1][:,i], time[i])
                else:
                    yield (*data, self._find_time())
            else:
                raise NotImplementedError()

    @property
    def ignore_sanity(self):
        return True
