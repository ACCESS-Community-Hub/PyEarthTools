# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations

import numpy as np
import warnings

from edit.pipeline_V2 import PipelineMod, PipelineWarning


class CustomLoader(PipelineMod):
    """
    DataLoader to batch data
    """

    def __init__(self, batch_size: int):
        """Custom DataLoader to batch data up

        Args:
            batch_size (int):
                Batch size

        Raises:
            TypeError:
                If `batch_size` is not an int
        """
        super().__init__()
        self.record_initialisation()

        if not isinstance(batch_size, int):
            raise TypeError(f"'batch_size' must be int, not {type(batch_size)}")
        self.batch_size = batch_size

        self._info_ = dict(batch_size=batch_size)

    def __getitem__(self, idx):
        return self.parent_pipeline()[idx]

    def __iter__(self):
        iterator = iter(self.parent_pipeline())

        batch = []
        try:
            while True:
                batch.append(iterator.__next__())
                if len(batch) == self.batch_size:
                    yield tuple(map(np.stack, zip(*batch)))
                    batch = []

        except StopIteration:
            return tuple(map(np.stack, zip(*batch)))
