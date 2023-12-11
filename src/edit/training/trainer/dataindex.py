"""
Provide a Machine Learning Model as an [edit.data Index][edit.data.indexes].

This will allow data to be retrieved as normal, with the user not having to worry about it being an ML Model
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from edit.data import EDITDatetime, Transform, TransformCollection, TimeDelta
from edit.data.indexes import BaseCacheIndex, TimeIndex

import edit.training.trainer
from edit.training.trainer import from_yaml


class MLDataIndex(BaseCacheIndex, TimeIndex):
    def __init__(
        self,
        trainer: edit.training.trainer.EDIT_Inference,
        *,
        data_interval: tuple, 
        cache: str | Path = None,
        predict_config: dict = dict(undo=True),
        recurrent_config: dict = {},
        offsetInterval: bool | tuple | TimeDelta = False,
        post_transforms: Transform | TransformCollection = TransformCollection(),
        override: bool = False,
        **kwargs,
    ):
        """Setup ML Data Index from defined trainer

        !!! Info
            This can be used just like an [Index][edit.data.indexes] from [edit.data][edit.data],
            so calling or indexing into this object work, as well as supplying transforms.

        Args:
            trainer (EDITTrainer):
                EDITTrainer to use to retrieve data
            data_interval (tuple):
                Resolution that the trainer operates at, in `TimeDelta` form. 
                e.g. (1, 'day')
            cache (str | Path, optional):
                Location to cache outputs, if not supplied don't cache.
            predict_config (dict, optional):
                Configuration for standard prediction.
            recurrent_config (dict, optional):
                Configuration if model must be run recurrently
            offsetInterval (bool, optional):
                Whether to offset time by interval. Defaults to False.
            **kwargs (dict, optional):
                Any keyword arguments to pass to [DataIndex][edit.data.DataIndex]
        """
        super().__init__(cache=cache, data_interval = data_interval, **kwargs)
        self.trainer = trainer
        self.predict_config = predict_config
        self.recurrent_config = recurrent_config

        self.post_transforms = post_transforms

        self.offsetInterval = offsetInterval
        self.override = override

    def offset_time(self, time) -> EDITDatetime:
        time = EDITDatetime(time)
        if self.offsetInterval:
            if self.data_interval and isinstance(self.offsetInterval, bool):
                time = EDITDatetime(time) + self.data_interval
            else:
                time = EDITDatetime(time) + TimeDelta(self.offsetInterval)
        return time
    

    def generate(
        self,
        querytime: str | EDITDatetime,
    ):  # transforms: Union[Callable, TransformCollection, Transform]= None
        """
        Get Data from given timestep
        """
        querytime = self.offset_time(querytime)

        querytime = querytime.at_resolution(self.data_resolution)
        predictions = None
        if self.recurrent_config:
            predictions = self.trainer.recurrent(
                querytime, interval = self.recurrent_config.pop('interval', self.data_interval), **self.recurrent_config, quiet = True,
            )
        else:
            predictions = self.trainer.predict(querytime, interval = self.predict_config.pop('interval', self.data_interval), **self.predict_config, quiet = True)

        if isinstance(predictions, (list, tuple)):
            predictions = predictions[1]
        
        if hasattr(self, 'base_transforms'):
            predictions = self.base_transforms(predictions)
        predictions = self.post_transforms(predictions)

        self._save_catalog()

        return predictions

    def filesystem(self, *args, **kwargs) -> Path:
        if self.override:
            with self.toggle_generate:
                return super().filesystem(*args, **kwargs)
        return super().filesystem(*args, **kwargs)

    def input_data(self, querytime: str):
        """
        Get input data at given timestep
        """
        querytime = self.offset_time(querytime)

        input_data = self.trainer.pipeline.undo(
            self.trainer.pipeline[querytime]
        )
        return input_data

    @property
    def data(self):
        return self.trainer.pipeline

    @staticmethod
    def from_yaml(
        yaml_config: str | Path,
        data_interval: tuple,
        checkpoint_path: str | bool = True,
        *,
        only_state: bool = False,
        stride_override: int | None = None,
        **kwargs,
    ):
        """Setup ML Data Index from yaml file config and pretrained model

        Args:
            yaml_config (str | Path):
                Path to yaml config
            data_interval (tuple):
                Resolution that the trainer operates at, in `TimeDelta` form. 
                e.g. (1, 'day')
            checkpoint_path (str | bool, optional):
                Path to pretrained checkpoint. Defaults to True.
            only_state (bool, optional):
                Only load the state of the model. Defaults to False.
            stride_override (int, optional):
                Values to override stride with, if using `PatchingDataIndex`. Defaults to None.

        Raises:
            RuntimeError:
                If no `checkpoint_path` is given

        Returns:
            (MLDataIndex):
                MLDataIndex to use to get data with
        """
        trainer: EDITTrainer
        trainer = from_yaml(
            yaml_config,
            strategy=kwargs.pop("strategy", "auto"),
            logger=kwargs.pop("logger", False),
            **kwargs,
        )
        trainer.load(checkpoint_path, only_state=only_state)

        return MLDataIndex(trainer, data_interval = data_interval, stride_override=stride_override)

