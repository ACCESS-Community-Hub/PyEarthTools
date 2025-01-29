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


from __future__ import annotations
import functools

from typing import Literal, TypeVar, Any, Optional

from abc import abstractmethod

from pyearthtools.data.time import TimeDelta
import xarray as xr
import numpy as np
import tqdm.auto as tqdm

from pyearthtools.pipeline.controller import Pipeline
from pyearthtools.training.wrapper.wrapper import ModelWrapper
from pyearthtools.training.wrapper.predict.timeseries import TimeSeriesPredictor

from pyearthtools.training.manage import Variables

XR_TYPE = TypeVar("XR_TYPE", xr.Dataset, xr.DataArray)


class CoupledPredictionWrapper(TimeSeriesPredictor):
    """
    Coupled Prediction Wrapper
    """

    def __init__(
        self,
        model: ModelWrapper,
        reverse_pipeline: Pipeline | int | str | None = None,
        *,
        fix_time_dim: bool = True,
        interval: int | str | TimeDelta = 1,
        time_dim: str = "time",
    ):
        super().__init__(model, reverse_pipeline, fix_time_dim=fix_time_dim, interval=interval, time_dim=time_dim)
