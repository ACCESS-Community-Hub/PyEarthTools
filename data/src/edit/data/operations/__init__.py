# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

"""
Useful Data Operations to apply to [indexes][edit.data.indexes] or [Datasets][xarray.Dataset]

## [xarray][xarray] Operations
| Name        | Description |
| :---        |     ----:   |
| [Percentile][edit.data.operations.percentile.percentile]  |  Find Percentiles of Data    |
| [Aggregation][edit.data.operations.aggregation.aggregation]  |  Aggregate Data across or leaving dims   |
| [Spatial Interpolation][edit.data.operations.interpolation.SpatialInterpolation]  |  Spatially Interpolate Datasets together    |
| [Temporal Interpolation][edit.data.operations.interpolation.TemporalInterpolation]  |  Temporally Interpolate Datasets together    |

## [Index][edit.data.indexes] Operations
| Name        | Description |
| :---        |     ----:   |
| [Series Indexing][edit.data.operations.index_routines.series]  |  Get a series of Data    |
| [Safe Series Indexing][edit.data.operations.index_routines.safe_series]  |  Safely get a series of Data    |
"""

from edit.data.operations import interpolation
from edit.data.operations.interpolation import (
    SpatialInterpolation,
    TemporalInterpolation,
    FullInterpolation,
)
from edit.data.operations.percentile import percentile
from edit.data.operations.aggregation import aggregation
from edit.data.operations.binning import binning

# from edit.data.operations.index_routines import safe_series, series
