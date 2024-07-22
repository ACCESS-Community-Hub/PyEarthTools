# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

# ruff: noqa: F401

"""
Prediction Wrappers
"""

from edit.utils.decorators import BackwardsCompatibility

from edit.training.wrapper.predict.predict import Predictor
from edit.training.wrapper.predict.timeseries import (
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