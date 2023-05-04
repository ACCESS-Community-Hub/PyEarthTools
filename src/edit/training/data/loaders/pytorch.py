from __future__ import annotations

from torch.utils.data import IterableDataset

from edit.training.data.templates import DataStep, DataIterator
from edit.training.data.sequential import SequentialIterator


@SequentialIterator
class PytorchIterable(DataStep, IterableDataset):
    """
    Connect Data Pipeline with PyTorch IterableDataset

    !!! Example
        ```python
        PytorchIterable(PipelineStep)

        ## As this is decorated with @SequentialIterator, it can be partially initialised

        partialPytorchIterable = PytorchIterable()
        partialPytorchIterable(PipelineStep)
        ```
    """

    def __init__(self, index: DataStep | DataIterator) -> None:
        super().__init__(index=index)

    def __getitem__(self, idx):
        return self.index[idx]

    def __iter__(self):
        for i in self.index:
            yield i

    @property
    def ignore_sanity(self):
        return True
