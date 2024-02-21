"""
Provide a Machine Learning Model as an [edit.data Index][edit.data.indexes].

This will allow data to be retrieved as normal, with the user not having to worry about it being an ML Model
"""
from __future__ import annotations

from pathlib import Path
import yaml
from typing import Any

import edit.data
from edit.data import EDITDatetime, Transform, TransformCollection, TimeDelta
from edit.data.indexes import BaseCacheIndex, TimeIndex

import edit.training.trainer
from edit.training.trainer import from_yaml

ATTRIBUTE_MARK = edit.data.transform.attributes.set_attributes(
    purpose = "Research Use Only.",
    contact = "For further information or support, contact the Data Science and Emerging Technologies Team.",
    credit = "Generated with `edit`, a research endeavour under the DSET team, and Project 3.1.",
    apply_on = 'dataset',
    )

class MLDataIndex(BaseCacheIndex, TimeIndex):
    def __init__(
        self,
        trainer: edit.training.trainer.EDIT_Inference,
        *,
        data_interval: tuple, 
        cache: str | Path | None = None,
        predict_config: dict[str, Any] | None = None,
        recurrent_config: dict[str, Any] | None = None,
        offsetInterval: bool | tuple | TimeDelta = False,
        post_transforms: Transform | TransformCollection | None = None,
        override: bool = False,
        data_attributes: str | Path | None = None,
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
            post_transforms (Transform | TransformCollection | None, optional):
                Transforms to apply post generation. Defaults to None.
            override (bool, optional):
                Override any generated data. Defaults to False.
            data_attributes (str | Path | None, optional):
                Path to yaml file specifying attributes to set.
            **kwargs (dict, optional):
                Any keyword arguments to pass to [BaseCacheIndex][edit.data.BaseCacheIndex]
        """
        super().__init__(cache=cache, **dict(kwargs))

        self.set_interval(data_interval)

        self.trainer = trainer

        if predict_config is None:
            predict_config = {}
        self.predict_config = dict(predict_config)

        if recurrent_config is None:
            recurrent_config = {}
        self.recurrent_config = dict(recurrent_config)

        if post_transforms is None:
            post_transforms = TransformCollection()
        self.post_transforms = post_transforms

        self.data_attributes = data_attributes

        self.offsetInterval = offsetInterval
        self.to_override = override

    def offset_time(self, time: str | EDITDatetime) -> EDITDatetime:
        """
        Offset the time given

        Controlled by how the init args are set.
        If `offsetInterval` is a bool and True, offset by interval
        Otherwise offset by `offsetInterval`.

        Args:
            time (str | EDITDatetime): 
                Time to offset

        Returns:
            (EDITDatetime): 
                Offset time
        """
        time = EDITDatetime(time)
        if self.offsetInterval:
            if self.data_interval and isinstance(self.offsetInterval, bool):
                time = EDITDatetime(time) + self.data_interval
            else:
                time = EDITDatetime(time) + TimeDelta(self.offsetInterval)
        return EDITDatetime(time)
    

    def generate(
        self,
        querytime: str | EDITDatetime,
    ) -> Any: 
        """
        Get Data from given timestep
        """
        querytime = self.offset_time(querytime)

        if self.data_resolution is not None:
            querytime = querytime.at_resolution(self.data_resolution)

        predictions = None

        if self.recurrent_config:
            predictions = self.trainer.recurrent(
                querytime, **self.recurrent_config,
            )
        else:
            predictions = self.trainer.predict(querytime, **self.predict_config)

        if isinstance(predictions, (list, tuple)):
            predictions = predictions[1]
        
        if hasattr(self, 'base_transforms'):
            predictions = self.base_transforms(predictions)
            
        predictions = self.post_transforms(predictions)
        predictions = ATTRIBUTE_MARK(predictions)
        
        if self.data_attributes is not None:
            attrs = yaml.safe_load(str(self.data_attributes))
            predictions = edit.data.transform.attributes.set_attributes(attrs, apply_on = 'dataset')(predictions)
            
        return predictions

    def filesystem(self, *args, **kwargs) -> Path | dict[str, str | Path] | list[str | Path]:
        if self.to_override:
            with self.override:
                return super().filesystem(*args, **kwargs)
        return super().filesystem(*args, **kwargs)

    @property
    def data(self):
        """Get Data Pipeline"""
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
        trainer = from_yaml(
            yaml_config,
            strategy=kwargs.pop("strategy", "auto"),
            logger=kwargs.pop("logger", False),
            **kwargs,
        )
        trainer.load(checkpoint_path, only_state=only_state)

        return MLDataIndex(trainer, data_interval = data_interval, stride_override=stride_override)

