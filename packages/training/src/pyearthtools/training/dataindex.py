# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Provide a Machine Learning Model as an [pyearthtools.data Index][pyearthtools.data.indexes].

This will allow data to be retrieved as normal, with the user not having to worry about it being an ML Model
"""
from __future__ import annotations

import os

from pathlib import Path
import yaml
from typing import Any

import pyearthtools.data
from pyearthtools.data import Petdt, Transform, TransformCollection, TimeDelta
from pyearthtools.data.indexes import TimeIndex
from pyearthtools.data.indexes.cacheIndex import BaseCacheIndex

import pyearthtools.training.wrapper

ATTRIBUTE_MARK = pyearthtools.data.transforms.attributes.set_attributes(
    purpose="Research Use Only.",
    contact="For further information or support, contact the Data Science and Emerging Technologies Team.",
    crpyearthtools="Generated with `pyearthtools`, a research endeavour under the DSET team, and Project 3.1.",
    apply_on="dataset",
)


class MLDataIndex(BaseCacheIndex, TimeIndex):
    """
    `pyearthtools.training` DataIndex

    Uses an underlying ML model to generate data to cache.
    """

    _save_self = False

    def __init__(
        self,
        wrapper: pyearthtools.training.wrapper.Predictor,
        *,
        data_interval: tuple,
        cache: os.PathLike | None = None,
        prediction_function: str = "predict",
        prediction_config: dict[str, Any] | None = None,
        offsetInterval: bool | tuple | TimeDelta = False,
        post_transforms: Transform | TransformCollection | None = None,
        override: bool = False,
        data_attributes: os.PathLike | None = None,
        **kwargs,
    ):
        """Setup ML Data Index from defined wrapper

        Info:
            This can be used just like an [Index][pyearthtools.data.indexes] from [pyearthtools.data][pyearthtools.data],
            so calling or indexing into this object work, as well as supplying transforms.

        Args:
            wrapper (pyearthtoolsTrainer):
                pyearthtoolsTrainer to use to retrieve data
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
                Any keyword arguments to pass to [BaseCacheIndex][pyearthtools.data.BaseCacheIndex]
        """

        super().__init__(cache=str(cache), **dict(kwargs))
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
        self._override = override

    def offset_time(self, time: str | Petdt) -> Petdt:
        """
        Offset the time given

        Controlled by how the init args are set.
        If `offsetInterval` is a bool and True, offset by interval
        Otherwise offset by `offsetInterval`.

        Args:
            time (str | Petdt):
                Time to offset

        Returns:
            (Petdt):
                Offset time
        """
        time = Petdt(time)
        if self.offsetInterval:
            if self.data_interval and isinstance(self.offsetInterval, bool):
                time = Petdt(time) + (
                    self.data_interval if not isinstance(self.data_interval, str) else TimeDelta(self.data_interval)
                )
            else:
                time = Petdt(time) + TimeDelta(self.offsetInterval)
        return Petdt(time)

    def _generate(
        self,
        querytime: str | Petdt,
    ) -> Any:
        """
        1. Generate the prediction from the model
        2. Apply post-model transforms
        3. Return the transformed predictions
        """
        querytime = self.offset_time(querytime)

        if self.data_resolution is not None:
            querytime = querytime.at_resolution(self.data_resolution)

        predictions = getattr(self.wrapper, self.prediction_function)(querytime, **self.predict_config)

        # TODO: Figure out why we need to do this and document it
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[1]

        # Apply base transforms to the prediction data
        if hasattr(self, "base_transforms"):
            predictions = self.base_transforms(predictions)

        # Apply post transforms to the predictions
        predictions = self.post_transforms(predictions)
        predictions = ATTRIBUTE_MARK(predictions)

        # Apply attribute marks to the prediction data
        if self.data_attributes is not None:
            attrs = yaml.safe_load(open(str(self.data_attributes), "r"))
            predictions = pyearthtools.data.transforms.attributes.set_attributes(attrs, apply_on="dataset")(predictions)

        return predictions

    def filesystem(self, *args, **kwargs) -> Path | dict[str, str | Path] | list[str | Path]:
        return super().filesystem(*args, **kwargs)

    def get(self, *args, **kwargs):
        """
        Retrieve the prediction data for the request sought
        """

        dt_sought = args[0]

        data = self._generate(dt_sought)
        return data

    @property
    def data(self):
        """Get Data Pipeline"""
        return self.wrapper.pipelines
