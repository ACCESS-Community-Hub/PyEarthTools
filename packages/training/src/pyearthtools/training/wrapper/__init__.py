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


from pyearthtools.training.wrapper.wrapper import ModelWrapper

from pyearthtools.training.wrapper import predict, train, utils

from pyearthtools.training.wrapper.train import TrainingWrapper
from pyearthtools.training.wrapper.predict import Predictor

try:
    ONNX_IMPORTED = True
    from pyearthtools.training.wrapper import onnx
except (ImportError, ModuleNotFoundError):
    ONNX_IMPORTED = False

try:
    LIGHTNING_IMPORTED = True
    from pyearthtools.training.wrapper import lightning
except (ImportError, ModuleNotFoundError):
    LIGHTNING_IMPORTED = False

__all__ = ["ModelWrapper", "predict", "train", "utils", "TrainingWrapper", "Predictor"]

if ONNX_IMPORTED:
    __all__.append("onnx")

if LIGHTNING_IMPORTED:
    __all__.append("lightning")
