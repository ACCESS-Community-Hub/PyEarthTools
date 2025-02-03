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


# ruff: noqa: F401

"""
Prediction Wrappers
"""

from pyearthtools.utils.decorators import BackwardsCompatibility

from pyearthtools.training.wrapper.predict.predict import Predictor
from pyearthtools.training.wrapper.predict.timeseries import (
    TimeSeriesPredictor,
    TimeSeriesAutoRecurrentPredictor,
    TimeSeriesManagedPredictor,
    ManualTimeSeriesPredictor,
)


@BackwardsCompatibility(TimeSeriesPredictor)
def TimeSeriesPredictionWrapper(): ...


@BackwardsCompatibility(TimeSeriesAutoRecurrentPredictor)
def TimeSeriesAutoRecurrent(): ...


@BackwardsCompatibility(TimeSeriesManagedPredictor)
def TimeSeriesManagedRecurrent(): ...


@BackwardsCompatibility(ManualTimeSeriesPredictor)
def ManualTimeSeriesPredictionWrapper(): ...
