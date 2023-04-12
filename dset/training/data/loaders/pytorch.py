from typing import Union
from torch.utils.data import IterableDataset

from dset.training.data.templates import DataStep, SequentialIterator, DataIterator

@SequentialIterator
class PytorchIterable(DataStep, IterableDataset):
    """
    Connect Data Pipeline with PyTorch IterableDataset
    """
    def __init__(self, index: Union[DataStep, DataIterator]) -> None:
        super().__init__(index = index)

    def __getitem__(self, idx):
        return self.index[idx]

    def __iter__(self):
        for i in self.index:
            yield i

    @property
    def ignore_sanity(self):
        return True