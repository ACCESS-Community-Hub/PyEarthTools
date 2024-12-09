# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

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
