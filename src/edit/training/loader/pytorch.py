from __future__ import annotations

from torch.utils.data import IterableDataset

from edit.pipeline.templates import DataStep, DataIterator
from edit.pipeline.sequential import SequentialDecorator


@SequentialDecorator
class PytorchIterable(DataStep, IterableDataset):
    """
    Connect Data Pipeline with PyTorch IterableDataset

    !!! Example
        ```python
        PytorchIterable(PipelineStep)

        ## As this is decorated with @SequentialDecorator, it can be partially initialised

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
    def ignore_debug(self):
        return True
