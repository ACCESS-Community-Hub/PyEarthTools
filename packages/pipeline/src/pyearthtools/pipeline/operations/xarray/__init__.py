# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

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
