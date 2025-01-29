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
Useful Data Operations to apply to [indexes][pyearthtools.data.indexes] or [Datasets][xarray.Dataset]

## [xarray][xarray] Operations
| Name        | Description |
| :---        |     ----:   |
| [Percentile][pyearthtools.data.operations.percentile.percentile]  |  Find Percentiles of Data    |
| [Aggregation][pyearthtools.data.operations.aggregation.aggregation]  |  Aggregate Data across or leaving dims   |
| [Spatial Interpolation][pyearthtools.data.operations.interpolation.SpatialInterpolation]  |  Spatially Interpolate Datasets together    |
| [Temporal Interpolation][pyearthtools.data.operations.interpolation.TemporalInterpolation]  |  Temporally Interpolate Datasets together    |

## [Index][pyearthtools.data.indexes] Operations
| Name        | Description |
| :---        |     ----:   |
| [Series Indexing][pyearthtools.data.operations.index_routines.series]  |  Get a series of Data    |
| [Safe Series Indexing][pyearthtools.data.operations.index_routines.safe_series]  |  Safely get a series of Data    |
"""

from pyearthtools.data.operations import interpolation
from pyearthtools.data.operations.interpolation import (
    SpatialInterpolation,
    TemporalInterpolation,
    FullInterpolation,
)
from pyearthtools.data.operations.percentile import percentile
from pyearthtools.data.operations.aggregation import aggregation
from pyearthtools.data.operations.binning import binning

# from pyearthtools.data.operations.index_routines import safe_series, series
