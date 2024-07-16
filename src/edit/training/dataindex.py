# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

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

import edit.training.wrapper

ATTRIBUTE_MARK = edit.data.transforms.attributes.set_attributes(
    purpose="Research Use Only.",
    contact="For further information or support, contact the Data Science and Emerging Technologies Team.",
    credit="Generated with `edit`, a research endeavour under the DSET team, and Project 3.1.",
    apply_on="dataset",
)


class MLDataIndex(BaseCacheIndex, TimeIndex):
    """
    `edit.training` DataIndex

    Uses an underlying ML model to generate data to cache.
    """

    _save_self = False

    def __init__(
        self,
        wrapper: edit.training.wrapper.PredictionWrapper,
        *,
        data_interval: tuple,
        cache: str | Path | None = None,
        prediction_function: str = "predict",
        prediction_config: dict[str, Any] | None = None,
        offsetInterval: bool | tuple | TimeDelta = False,
        post_transforms: Transform | TransformCollection | None = None,
        override: bool = False,
        data_attributes: str | Path | None = None,
        **kwargs,
    ):
        """Setup ML Data Index from defined wrapper

        Info:
            This can be used just like an [Index][edit.data.indexes] from [edit.data][edit.data],
            so calling or indexing into this object work, as well as supplying transforms.

        Args:
            wrapper (EDITTrainer):
                EDITTrainer to use to retrieve data
            data_interval (tuple):
                Resolution that the wrapper operates at, in `TimeDelta` form.
                e.g. (1, 'day')
            cache (str | Path, optional):
                Location to cache outputs, if not supplied don't cache.
            prediction_function (str, optional):
                Function to use for prediction
            prediction_config (dict, optional):
                Configuration if predictions
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
        self.record_initialisation()

        self.set_interval(data_interval)

        self.wrapper = wrapper

        self.predict_config = dict(prediction_config or {})
        self.prediction_function = prediction_function

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
                time = EDITDatetime(time) + (
                    self.data_interval if not isinstance(self.data_interval, str) else TimeDelta(self.data_interval)
                )
            else:
                time = EDITDatetime(time) + TimeDelta(self.offsetInterval)
        return EDITDatetime(time)

    def _generate(
        self,
        querytime: str | EDITDatetime,
    ) -> Any:
        """
        Get Data from given timestep
        """
        querytime = self.offset_time(querytime)

        if self.data_resolution is not None:
            querytime = querytime.at_resolution(self.data_resolution)

        predictions = getattr(self.wrapper, self.prediction_function)(querytime, **self.predict_config)

        if isinstance(predictions, (list, tuple)):
            predictions = predictions[1]

        if hasattr(self, "base_transforms"):
            predictions = self.base_transforms(predictions)

        predictions = self.post_transforms(predictions)
        predictions = ATTRIBUTE_MARK(predictions)

        if self.data_attributes is not None:
            attrs = yaml.safe_load(open(str(self.data_attributes), "r"))
            predictions = edit.data.transforms.attributes.set_attributes(attrs, apply_on="dataset")(predictions)
        return predictions

    def filesystem(self, *args, **kwargs) -> Path | dict[str, str | Path] | list[str | Path]:
        if self.to_override:
            with self.override:
                return super().filesystem(*args, **kwargs)
        return super().filesystem(*args, **kwargs)

    @property
    def data(self):
        """Get Data Pipeline"""
        return self.wrapper.pipelines
