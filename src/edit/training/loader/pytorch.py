# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations

from torch.utils.data import IterableDataset, get_worker_info

from edit.pipeline_V2 import Pipeline


class PytorchIterable(IterableDataset):
    """
    Connect Data Pipeline with PyTorch IterableDataset

    """

    def __init__(self, pipeline: Pipeline) -> None:
        super().__init__()
        self._pipeline = pipeline

    def save(self, *args, **kwargs):
        return self._pipeline.save(*args, **kwargs)

    def __iter__(self):
        pipeline = self._pipeline
        samples = [t for t in pipeline.iteration_order]
        worker_info = get_worker_info()

        if worker_info is None:  # single-process data loading, return the full iterator
            for i in samples:
                data = pipeline.get_and_catch[i]
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

                data = pipeline.get_and_catch[samples[i]]
                if data is None:
                    continue
                yield data
        self._current_index = None

    @property
    def ignore_debug(self):
        return True
