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
pyearthtools Utilities
"""

__version__ = "0.2.0"

import importlib
import importlib.util

import pyearthtools
from pyearthtools.utils import (
    config,
    context,
    data,
    decorators,
    initialisation,
    logger,
    parameter,
    repr_utils,
)
from pyearthtools.utils.initialisation import dynamic_import, load, save

setattr(pyearthtools, "config", config)
