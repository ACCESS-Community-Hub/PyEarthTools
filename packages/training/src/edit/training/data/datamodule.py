# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
Training DataModule to source data from `pipeline`'s.
"""

from __future__ import annotations

import functools
from typing import Optional, Callable, Any, TypeVar

import numpy as np
from pathlib import Path

import pyearthtools.pipeline
from pyearthtools.pipeline import Pipeline, Iterator
from pyearthtools.utils.initialisation import InitialisationRecordingMixin


CONFIG_KEY = "--CONFIG--"
SUFFIX = ".datamodule"

T = TypeVar("T", Any, Any)
R = TypeVar("R", Any, Any)


def load_pipelines(pipeline: Pipeline | str) -> Pipeline:
    """Load pipelines if str"""
    if isinstance(pipeline, str):
        return pyearthtools.pipeline.load(pipeline)
    return pipeline


class PipelineDataModule(InitialisationRecordingMixin):
    """
    Base `PipelineDataModule`

    `get_sample` can be used to retrieve from `pipelines`, and `fake_batch_dim` can be overriden if
    special batch faking is needed.

    `train` configures the pipelines from `train_split` and `valid` for validation.


    """

    _train: Optional[bool] = None

    def __init__(
        self,
        pipelines: dict[str, str | Pipeline | tuple[Pipeline, ...]] | tuple[Pipeline | str, ...] | Pipeline | str,
        train_split: Optional[Iterator] = None,
        valid_split: Optional[Iterator] = None,
    ):
        """
        Setup `Pipeline`'s for use with ML Training

        Args:
            pipelines (dict[str, str | Pipeline | tuple[Pipeline, ...]] | tuple[Pipeline | str, ...] | Pipeline | str):
                Pipelines for data retrieval, can be dictionary and/or list/tuple of `Pipelines` or a single `Pipeline`
            train_split (Optional[Iterator], optional):
                Iterator to use for training. Pipelines configured by calling `.train()`. Defaults to None.
            valid_split (Optional[Iterator], optional):
                Iterator to use for validation. Pipelines configured by calling `.valid()`. Defaults to None.
        """
        super().__init__()
        self.record_initialisation()

        if isinstance(pipelines, list):
            pipelines = tuple(pipelines)
        if isinstance(pipelines, (tuple, list)) and len(pipelines) == 1:
            pipelines = pipelines[0]

        self._pipelines = self.map_function(pipelines, load_pipelines)

        self.update_initialisation(pipelines=self._pipelines)

        self._train_split = train_split
        self._valid_split = valid_split

    @property
    def pipelines(self) -> dict[str, Pipeline | tuple[Pipeline, ...]] | tuple[Pipeline, ...] | Pipeline:
        return self._pipelines  # type: ignore

    @classmethod
    def map_function(
        cls, obj: dict[str, T | tuple[T, ...]] | tuple[T, ...] | T, function: Callable[[Any], R], **kwargs: Any
    ) -> dict[str, R | tuple[R, ...]] | tuple[R, ...] | R:
        recur_function = functools.partial(PipelineDataModule.map_function, function=function, **kwargs)
        if isinstance(obj, dict):
            return {key: recur_function(val) for key, val in obj.items()}  # type: ignore
        if isinstance(obj, (list, tuple)):
            return type(obj)(map(recur_function, obj))  # type: ignore
        return function(obj, **kwargs)

    def map_function_to_pipelines(self, function: Callable[[Pipeline], Any], **kwargs):
        """
        Map a function over `Pipelines`
        """
        return self.map_function(self.pipelines, function, **kwargs)

    def train(self):
        """
        Set `Pipeline`s to iterate over `train_split`
        """
        if self._train_split is None:
            raise ValueError("Cannot enter training mode as `train_split` is None.")

        self._train = True

        def set_iterator(obj: Pipeline):
            obj.iterator = self._train_split

        self.map_function_to_pipelines(set_iterator)

    def eval(self):
        """
        Set `Pipeline`s to iterate over `valid_split`
        """
        if self._valid_split is None:
            raise ValueError("Cannot enter validation mode as `valid_split` is None.")

        self._train = False

        def set_iterator(obj: Pipeline):
            obj.iterator = self._valid_split

        self.map_function_to_pipelines(set_iterator)

    def check_for_use(self):
        """Check if `datamodule` is ready for use."""
        if self._train is None:
            raise RuntimeError(
                "An iterator has not be configured, call either `.train()`, or `.eval()` for training / evaluation mode."
            )

    def __getitem__(self, idx):
        """
        Index into datamodule

        Converts int `idx` into iterators idx'nth sample.
        i.e. idx = 1
            >>> train_split.samples[idx]
        """
        self.check_for_use()

        iterator = self._train_split if self._train else self._valid_split or self._train_split

        if iterator is None:
            raise TypeError("Cannot index into data without an iterator set.")

        idx = iterator.samples[idx]
        return self.map_function_to_pipelines(lambda x: x[idx])

    def __iter__(self):
        self.check_for_use()

        generators = self.map_function_to_pipelines(iter)
        while True:
            try:
                yield self.map_function(generators, next)
            except StopIteration:
                break

    def fake_batch_dim(self, sample):
        """Fake batch dim on `sample`"""
        return np.expand_dims(sample, 0)

    def get_sample(self, idx, *, fake_batch_dim: bool = False):
        """Get sample from `pipeline`s"""
        if fake_batch_dim:

            def add_batch_dim(obj):
                if isinstance(obj, (list, tuple)):
                    return type(obj)(map(add_batch_dim, obj))
                return self.fake_batch_dim(obj)

            return self.map_function_to_pipelines(lambda x: add_batch_dim(x[idx]))
        return self.map_function_to_pipelines(lambda x: x[idx])

    @classmethod
    def find_shape(cls, obj):
        """Find shape of `obj`"""
        if isinstance(obj, dict):
            return {key: PipelineDataModule.find_shape(val) for key, val in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(map(PipelineDataModule.find_shape, obj))
        return obj.shape

    def save(self, path: Optional[str | Path] = None) -> None | str:
        """
        Save `PipelineDataModule`

        Args:
            path (Optional[str | Path], optional):
                File to save to. If not given return save str. Defaults to None.

        Returns:
            (Union[None, str]):
                If `path` is None, `PipelineDataModule` in save form else None.
        """
        from pyearthtools.training.data.fileio import save

        return save(self, path)

    @classmethod
    def load(cls, stream: str | Path, **kwargs: Any) -> "PipelineDataModule":
        """
        Load `PipelineDataModule` config

        Args:
            stream (str | Path):
                File or dump to load
            kwargs (Any):
                Updates to default values include in the config.

        Returns:
            (PipelineDataModule):
                Loaded PipelineDataModule
        """
        from pyearthtools.training.data.fileio import load

        return load(stream, **kwargs)
