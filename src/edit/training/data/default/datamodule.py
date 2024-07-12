# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations
from typing import Union

from edit.pipeline.controller import Pipeline
from edit.pipeline.iterators import Iterator

from edit.training.data.datamodule import PipelineDataModule
from edit.training.data.default.datasets import IndexableDataset, IterableDataset


class DataLoader:
    def __init__(
        self, dataset: Union[IndexableDataset, IterableDataset], *, batch_size: int = 1, shuffle: bool = False
    ):
        raise NotImplementedError()
        pass

    def __iter__(self): ...

    def __len__(self): ...


class PipelineDefaultDataModule(PipelineDataModule):
    """
    Default dataloader, mimics the `Lightning` Datamodule
    """

    def __init__(
        self,
        pipelines: dict[str, Pipeline | tuple[Pipeline, ...]] | tuple[Pipeline, ...] | Pipeline,
        train_split: Iterator | None = None,
        valid_split: Iterator | None = None,
        iterator_dataset: bool = False,
        **kwargs,
    ):
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
        return self.map_function(self._dataloader, DataLoader, **self._kwargs)

    def valid_dataloader(self):
        self.eval()
        return self.map_function(self._dataloader, DataLoader, **self._kwargs)

    def predict_dataloader(self):
        self.eval()
        return self.map_function(self._dataloader, DataLoader, **self._kwargs)
