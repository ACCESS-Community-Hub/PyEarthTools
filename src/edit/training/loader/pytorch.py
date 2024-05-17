# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations
import math

from torch.utils.data import IterableDataset, get_worker_info

from edit.pipeline.templates import DataStep, DataIterator
from edit.pipeline.sequential import SequentialDecorator


@SequentialDecorator
class PytorchIterable(DataIterator, IterableDataset):
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

    def validate(self) -> bool:
        super_validate = super().validate()
        return super_validate and "ToNumpy" in self.steps

    def __iter__(self):
        samples = [t for t in self.generator]
        worker_info = get_worker_info()

        if worker_info is None:  # single-process data loading, return the full iterator
            for i in samples:
                data = self.get_catch(i)
                if data is None:
                    continue
                yield data

        else:  # in a worker process
            # split workload
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            for i in range(len(samples)):
                if not i % num_workers == worker_id:
                    continue
                if i >= len(samples):
                    continue
                self._current_index = samples[i]

                data = self.get_catch(samples[i])
                if data is None:
                    continue
                yield data
        self._current_index = None

    @property
    def ignore_debug(self):
        return True
