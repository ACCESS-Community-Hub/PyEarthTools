from typing import Union
import warnings
import torch

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
        super().__init__(index=index, apply_func=None, undo_func=self._undo_func)
        warnings.warn(f"ClimaX Dataloader will be moved out shortly")
        self._size = 2

    def _undo_func(self, data):
        return data[:self._size]

    def _find_time(self):
        if not hasattr(self.index, "sample_interval"):
            raise RuntimeError(f"Not using a known TemporalIterator")

        # if isinstance(data, tuple)

        return torch.Tensor([1])

    def __getitem__(self, idx):
        self._size = len(idx)
        data = self.index[idx]
        extend = data[0].shape[0]
        return (*data, self._find_time().expand(extend))

    def __iter__(self):
        for i in self.index:
            yield (*i, self._find_time())

    @property
    def ignore_sanity(self):
        return True
