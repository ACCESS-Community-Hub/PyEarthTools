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
Dask operations

| Category | Description | Available |
| -------- | ----------- | --------- |
| augument | Augument numpy data | `Rotate`, `Flip`, `Transform` | 
| Compute  | Call compute on an dask object | `Compute` |
| conversion | Convert between data types | `ToXarray`, `ToNumpy` |
| filters | Filter data when iterating | `DropAnyNan`, `DropAllNan`, `DropValue`, `Shape` |
| join | Combine tuples of `np.ndarrays` | `Stack`, `VStack`, `HStack`, `Concatenate` |
| normalisation | Normalise arrays | `Anomaly`, `Deviation`, `Division`, `Evaluated`  |
| reshape | Reshape numpy array | `Rearrange`, `Squish`, `Expand`, `Flatten`, `SwapAxis` |
| select | Select elements from array | `Select`, `Slice` |
| split  | Split numpy arrays into tuples | `OnAxis`, `OnSlice`, `VSplit`, `HSplit` |
| values | Modify values of arrays | `FillNan`, `MaskValue`, `ForceNormalised` |
"""


from pyearthtools.pipeline.operations.dask.join import Stack, Concatenate, VStack, HStack

from pyearthtools.pipeline.operations.dask.compute import Compute

from pyearthtools.pipeline.operations.dask import (
    augment,
    filters,
    normalisation,
    reshape,
    select,
    split,
    values,
    conversion,
)

__all__ = [
    "Stack",
    "Concatenate",
    "VStack",
    "HStack",
    "Compute",
    "augment",
    "filters",
    "reshape",
    "select",
    "split",
    "values",
    "normalisation",
    "conversion",
]
