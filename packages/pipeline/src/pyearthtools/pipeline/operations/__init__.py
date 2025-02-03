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
Pipeline Operations

| SubModules | Info |
| ---------- | ---- |
| numpy | Numpy arrays |
| xarray | Xarray |
| dask   | Dask arrays |
| transform   | Transformations |
"""

import warnings

from pyearthtools.pipeline.operations import xarray, numpy
from pyearthtools.pipeline.operations.transforms import Transforms
from pyearthtools.pipeline.operations import transform

try:
    from pyearthtools.pipeline.operations import dask
except (ImportError, ModuleNotFoundError) as e:
    warnings.warn(f"Unable to import `operations.dask` due to {e}", ImportWarning)

__all__ = [
    "xarray",
    "numpy",
    "transform",
    "Transforms",
    "dask",
]
