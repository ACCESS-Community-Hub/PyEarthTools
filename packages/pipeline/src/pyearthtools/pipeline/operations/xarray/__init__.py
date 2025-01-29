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
xarray Operations

| Category | Description | Available |
| -------- | ----------- | --------- |
| Compute  | Call compute on an xarray object | `Compute` |
| Chunk  | Rechunk xarray object | `Chunk` |
| conversion | Convert datasets between numpy or dask arrays | `ToNumpy`, `ToDask` |
| filters | Filter data when iterating | `DropAnyNan`, `DropAllNan`, `DropValue`, `Shape` |
| join | Join tuples of xarray objects | `Merge`, `Concatenate` |
| metadata | Modify or keep metadata | `Rename`, `Encoding`, `MaintainEncoding`, `Attributes`, `MaintainAttributes` |
| normalisation | Normalise datasets | `Anomaly`, `Deviation`, `Division`, `Evaluated` |
| reshape | Reshape datasets | `Dimension`, `CoordinateFlatten` |
| select | Select elements from dataset's | `SelectDataset`, `DropDataset`, `SliceDataset` |
| sort | Sort variables of a dataset | `Sort` |
| split | Split datasets | `OnVariables`, `OnCoordinate` |
| values | Modify values of datasets | `FillNan`, `MaskValue`, `ForceNormalised`, `Derive` |
| remapping | Reproject data | `HEALPix` | 
"""

from pyearthtools.pipeline.operations.xarray.compute import Compute
from pyearthtools.pipeline.operations.xarray.join import Merge, Concatenate
from pyearthtools.pipeline.operations.xarray.sort import Sort
from pyearthtools.pipeline.operations.xarray.chunk import Chunk

from pyearthtools.pipeline.operations.xarray import (
    conversion,
    filters,
    reshape,
    select,
    split,
    values,
    metadata,
    normalisation,
    remapping,
)

__all__ = [
    "Compute",
    "Merge",
    "Concatenate",
    "Sort",
    "Chunk",
    "conversion",
    "filters",
    "reshape",
    "select",
    "split",
    "values",
    "metadata",
    "normalisation",
    "remapping",
]
