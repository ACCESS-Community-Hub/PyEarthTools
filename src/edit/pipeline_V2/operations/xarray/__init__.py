# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
xarray Operations

| | Description | Available |
| - | --------- | --------- |
| conversion | Convert between datatypes | `ToNumpy`, `ToDask` |
| filters | Filter samples being iterated over | `DropAnyNan`, `DropAllNan`, `DropValue`, `Shape`  |

"""

from edit.pipeline_V2.operations.xarray.compute import Compute
from edit.pipeline_V2.operations.xarray.join import Merge, Concatenate
from edit.pipeline_V2.operations.xarray.sort import Sort

from edit.pipeline_V2.operations.xarray import (
    conversion,
    filters,
    reshape,
    select,
    split,
    values,
    metadata,
    normalisation,
)

__all__ = [
    "Compute",
    "Merge",
    "Concatenate",
    "Sort",
    "conversion",
    "filters",
    "reshape",
    "select",
    "split",
    "values",
    "metadata",
    "normalisation",
]
