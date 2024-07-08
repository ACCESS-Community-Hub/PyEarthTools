# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.


# ruff: noqa: F401


from edit.training.wrapper.wrapper import ModelWrapper

from edit.training.wrapper import predict, train, utils

from edit.training.wrapper.train import TrainingWrapper
from edit.training.wrapper.predict import PredictionWrapper

try:
    ONNX_IMPORTED = True
    from edit.training.wrapper import onnx
except (ImportError, ModuleNotFoundError):
    ONNX_IMPORTED = False

try:
    LIGHTNING_IMPORTED = True
    from edit.training.wrapper import lightning
except (ImportError, ModuleNotFoundError):
    LIGHTNING_IMPORTED = False

__all__ = ["ModelWrapper", "predict", "train", "utils", "TrainingWrapper", "PredictionWrapper"]

if ONNX_IMPORTED:
    __all__.append("onnx")

if LIGHTNING_IMPORTED:
    __all__.append("lightning")
