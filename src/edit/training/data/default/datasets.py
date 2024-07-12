# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations

from edit.pipeline import Pipeline


class BaseDefault:
    def __init__(self, pipeline: Pipeline, **kwargs) -> None:
        super().__init__(**kwargs)
        self._pipeline = pipeline

    def save(self, *args, **kwargs):
        return self._pipeline.save(*args, **kwargs)

    @property
    def iterator(self):
        return self._pipeline.iterator

    @iterator.setter
    def iterator(self, val):
        self._pipeline.iterator = val

    def __len__(self):
        return len(self._pipeline.iteration_order)


class IterableDataset(BaseDefault):
    """
    Iterate over pipeline
    """

    def __iter__(self):
        for sample in self._pipeline:
            yield sample


class IndexableDataset(BaseDefault):
    """Mapped Dataset of `Pipeline`"""

    def __getitem__(self, idx):
        return self._pipeline[self._pipeline.iteration_order[idx]]
