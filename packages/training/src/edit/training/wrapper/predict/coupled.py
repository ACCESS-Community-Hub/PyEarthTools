# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations
import functools

from typing import Literal, TypeVar, Any, Optional

from abc import abstractmethod

from edit.data.time import TimeDelta
import xarray as xr
import numpy as np
import tqdm.auto as tqdm

from edit.pipeline.controller import Pipeline
from edit.training.wrapper.wrapper import ModelWrapper
from edit.training.wrapper.predict.timeseries import TimeSeriesPredictor

from edit.training.manage import Variables

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
