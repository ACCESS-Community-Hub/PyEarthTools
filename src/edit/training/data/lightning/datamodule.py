# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations


import lightning as L
from torch.utils.data import DataLoader

from edit.pipeline.controller import Pipeline
from edit.pipeline.iterators import Iterator

from edit.training.data.datamodule import PipelineDataModule
from edit.training.data.lightning.datasets import PytorchDataset, PytorchIterable


class PipelineLightningDataModule(PipelineDataModule, L.LightningDataModule):
    """
    Pytorch Lightning DataModule.

    Wraps around `PipelineDataModule` to allow for usage with Lightning
    """
    def __init__(self, pipelines: dict[str, Pipeline | tuple[Pipeline, ...]] | tuple[Pipeline, ...] | Pipeline, train_split: Iterator | None = None, valid_split: Iterator | None = None, iterator_dataset:bool = False, **kwargs):
        super().__init__(pipelines, train_split, valid_split)
        self.record_initialisation()
        self._iterator_dataset = iterator_dataset
        self._kwargs = kwargs

        # def setup(self, stage = None):
        def make_torch_dataset(obj: Pipeline):
            if self._iterator_dataset:
                return PytorchIterable(obj)
            return PytorchDataset(obj)
        self._dataloader = self.map_function_to_pipelines(make_torch_dataset)

    def train_dataloader(self):
        self.train()
        return self.map_function(self._dataloader, DataLoader, **self._kwargs)
    
    def valid_dataloader(self):
        self.eval()
        return self.map_function(self._dataloader, DataLoader, **self._kwargs)
    
    def predict_dataloader(self):
        self.eval()
        return self.map_function(self._dataloader, DataLoader, **self._kwargs)