# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations


import lightning as L
from lightning.pytorch.utilities import CombinedLoader
from torch.utils.data import DataLoader

from edit.pipeline.controller import Pipeline
from edit.pipeline.iterators import Iterator

from edit.training.data.datamodule import PipelineDataModule
from edit.training.data.lightning.datasets import PytorchDataset, PytorchIterable, BasePytorchPipeline


class PipelineLightningDataModule(PipelineDataModule, L.LightningDataModule):
    """
    Pytorch Lightning DataModule.

    Wraps around `PipelineDataModule` to allow for usage with Lightning

    Usage:
        ```python
        datamodule = PipelineLightningDataModule(
            pipleines = Pipeline(...),
            train_split = edit.pipeline.iterators.DateRange('1980', '2020', '6 hours')
        )
        ```
    """

    def __init__(
        self,
        pipelines: dict[str, Pipeline | str | tuple[Pipeline, ...]] | tuple[Pipeline | str, ...] | Pipeline | str,
        train_split: Iterator | None = None,
        valid_split: Iterator | None = None,
        *,
        iterator_dataset: bool = False,
        **kwargs,
    ):
        """
        Create Pytorch lighting datamodule out of `PipelineDataModule`

        Args:
            pipelines (dict[str, Pipeline  |  tuple[Pipeline, ...]] | tuple[Pipeline, ...] | Pipeline):
                Pipelines to get data from
            train_split (Iterator | None, optional):
                Iterator defining training range. Defaults to None.
            valid_split (Iterator | None, optional):
                Iterator defining validation range. Defaults to None.
            iterator_dataset (bool, optional):
                Whether to use `PytorchIterable` or `PytorchDataset`. Defaults to False.
            kwargs:
                All kwargs passed to `torch.utils.DataLoader`.
                batch_size, num_workers, ...

        """
        super().__init__(pipelines, train_split, valid_split)
        self.record_initialisation()
        self.update_initialisation(pipelines=self.pipelines)

        self._iterator_dataset = iterator_dataset
        self._kwargs = kwargs

        # def setup(self, stage = None):
        def make_torch_dataset(obj: Pipeline) -> BasePytorchPipeline:
            if self._iterator_dataset:
                return PytorchIterable(obj.copy())
            return PytorchDataset(obj.copy())

        self._train_dataloader = self.map_function_to_pipelines(make_torch_dataset)
        self._valid_dataloader = self.map_function_to_pipelines(make_torch_dataset)

    def train_dataloader(self):
        self.train()
        return self.map_function(self._train_dataloader, DataLoader, **self._kwargs)

    def val_dataloader(self):
        self.eval()
        return CombinedLoader(self.map_function(self._valid_dataloader, DataLoader, **self._kwargs), "min_size")

    def train(self):
        """
        Set `Pipeline`s to iterate over `train_split`
        """
        if self._train_split is None:
            raise ValueError("Cannot enter training mode as `train_split` is None.")

        self._train = True

        def set_iterator(obj: BasePytorchPipeline):
            obj._pipeline.iterator = self._train_split  # type: ignore

        self.map_function(self._train_dataloader, set_iterator)

    def eval(self):
        """
        Set `Pipeline`s to iterate over `valid_split`
        """
        if self._valid_split is None:
            raise ValueError("Cannot enter validation mode as `valid_split` is None.")

        self._train = False

        def set_iterator(obj: BasePytorchPipeline):
            obj._pipeline.iterator = self._valid_split  # type: ignore

        self.map_function(self._valid_dataloader, set_iterator)
