from __future__ import annotations

import numpy as np
import warnings

from edit.pipeline.templates import DataStep, DataIterator
from edit.pipeline.sequential import SequentialDecorator

from edit.pipeline.warnings import PipelineWarning


@SequentialDecorator
class CustomLoader(DataStep):
    """
    DataLoader to batch data


    !!! Example
        ```python
        CustomLoader(PipelineStep, batch_size = 16)

        ## As this is decorated with @SequentialDecorator, it can be partially initialised

        partialCustomLoader = CustomLoader(batch_size = 16)
        partialCustomLoader(PipelineStep)
        ```
    """

    def __init__(self, index: DataStep, batch_size: int):
        """Custom DataLoader to batch data up

        Args:
            index (DataStep):
                Underlying DataStep to get data from
            batch_size (int):
                Batch size

        Raises:
            TypeError:
                If `batch_size` is not an int
        """
        super().__init__(index)

        if not self == self.step(-1):
            warnings.warn(f"{self} should be the last step in a DataPipeline.", PipelineWarning)

        if not isinstance(batch_size, int):
            raise TypeError(f"'batch_size' must be int, not {type(batch_size)}")
        self.batch_size = batch_size

        self._info_ = dict(batch_size=batch_size)

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
