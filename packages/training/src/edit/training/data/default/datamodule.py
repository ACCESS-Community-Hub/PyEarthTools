# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations
import functools
from typing import Callable, Any

import numpy as np
import xarray as xr

from edit.pipeline.controller import Pipeline
from edit.pipeline.iterators import Iterator

from edit.training.data.datamodule import PipelineDataModule
from edit.training.data.default.datasets import IndexableDataset, IterableDataset, BaseDefault


def map_function(obj, function: Callable[[Any], Any], **kwargs):
    """Map function over `obj`."""
    recur_function = functools.partial(map_function, function=function, **kwargs)
    if isinstance(obj, dict):
        return {key: recur_function(val) for key, val in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(map(recur_function, obj))
    return function(obj, **kwargs)


def combine_batches(samples):
    """Combine batches of data"""
    if all(map(lambda x: isinstance(x, np.ndarray), samples)):
        return np.stack(samples)
    if all(map(lambda x: isinstance(x, (xr.Dataset, xr.DataArray)), samples)):
        return xr.concat(samples, dim="batch").assign_coords(batch=range(len(samples)))
    raise TypeError(f"Cannot combine batches of type: {tuple(map(type, samples))}.")


class DataLoader:
    """
    Default Dataloader to allow batching of `default.datasets`.


    If combining xarray ensure coordinates are combinable,
    as in, if using a random iterator to get batches, a true time
    dimension will cause issues with missing values. Instead use
    a lead_time dimension which should be the same across batches.

    Usage:
        ```python
        dataloader = DataLoader(IndexableDataset(...), batch_size = 10)
        for i in dataloader:
            assert i.shape[0] == 10

        dataloader = DataLoader({'testing': IndexableDataset(...)}, batch_size = 10)
        for i in dataloader:
            i.keys() == ('testing',)
    """

    def __init__(
        self,
        dataset: BaseDefault | dict[str, BaseDefault] | tuple[BaseDefault, ...],
        *,
        batch_size: int = 1,
    ):
        """
        Default DataLoader to allow batch of datasets for training applications.

        Args:
            dataset (BaseDefault | dict[str, BaseDefault] | tuple[BaseDefault, ...]):
                Datasets to use. Can handle iterables of dict and tuples, and maintains upon iter.
            batch_size (int, optional):
                Number of elements to accumulate into a batch. Defaults to 1.
            ```
        """
        if not isinstance(dataset, BaseDefault):
            dataset = map_function(dataset, DataLoader, batch_size=batch_size)  # type: ignore
        self._dataset = dataset
        self.batch_size = batch_size

        self._index = 0
        self._generator = None

    def __next__(self):
        """
        Get next sample from `_dataset`.
        """
        if isinstance(self._dataset, BaseDefault):
            if isinstance(self._dataset, IndexableDataset):
                samples = tuple(
                    self._dataset[i] for i in range(self._index, min(self._index + self.batch_size, len(self)))
                )
                self._index += self.batch_size
            elif isinstance(self._dataset, IterableDataset):
                if self._generator is None:
                    self._generator = iter(self._dataset)
                samples = []
                try:
                    for _ in range(self.batch_size):
                        samples.append(next(self._generator))
                except StopIteration:
                    pass
                if len(samples) == 0:
                    raise StopIteration()
            return combine_batches(samples)
        return map_function(self._dataset, next)

    def __iter__(self):
        return self

    def __len__(self) -> int:
        def find_len(obj) -> int:
            if isinstance(obj, BaseDefault):
                return len(obj)
            elif isinstance(obj, (tuple, list)):
                return min(map(len, obj))
            elif isinstance(obj, dict):
                return min(map(find_len, obj.values()))
            raise TypeError(f"Cannot find length of {obj}.")

        return find_len(self._dataset)


class PipelineDefaultDataModule(PipelineDataModule):
    """
    Default dataloader, mimics the `Lightning` Datamodule

    Usage:
        ```python
        datamodule = PipelineDefaultDataModule(
            pipelines = Pipeline(...),
            train_split = edit.pipeline.iterators.DateRange('1980', '2020', '6 hours')
        )
    """

    def __init__(
        self,
        pipelines: dict[str, str | Pipeline | tuple[Pipeline, ...]] | tuple[Pipeline | str, ...] | Pipeline,
        train_split: Iterator | None = None,
        valid_split: Iterator | None = None,
        *,
        iterator_dataset: bool = False,
        **kwargs,
    ):
        """
        Default dataloader which mimics the `Lightning` datamodule,

        Args:
            pipelines (dict[str, str  |  Pipeline  |  tuple[Pipeline, ...]] | tuple[Pipeline  |  str, ...] | Pipeline):
                Pipelines for data retrieval, can be dictionary and/or list/tuple of `Pipelines` or a single `Pipeline`
            train_split (Iterator | None, optional):
                Iterator to use for training. Pipelines configured by calling `.train()`. Defaults to None.
            valid_split (Iterator | None, optional):
                Iterator to use for validation. Pipelines configured by calling `.valid()`. Defaults to None.
            iterator_dataset (bool, optional):
                Whether to use iterator dataset, which will iterate over pipeline instead of direct indexing. Defaults to False.
            kwargs:
                Keyword arguments to pass to `DataLoader`.
                e.g.: `batch_size`.

        """
        super().__init__(pipelines, train_split, valid_split)
        self.record_initialisation()

        self._iterator_dataset = iterator_dataset
        self._kwargs = kwargs

        def make_dataset(obj: Pipeline):
            if self._iterator_dataset:
                return IterableDataset(obj)
            return IndexableDataset(obj)

        self._dataloader = self.map_function_to_pipelines(make_dataset)

    def train_dataloader(self):
        """
        Get training dataloader
        """
        self.train()
        return DataLoader(self._dataloader, **self._kwargs)

    def valid_dataloader(self):
        self.eval()
        return DataLoader(self._dataloader, **self._kwargs)

    def predict_dataloader(self):
        self.eval()
        return DataLoader(self._dataloader, **self._kwargs)
