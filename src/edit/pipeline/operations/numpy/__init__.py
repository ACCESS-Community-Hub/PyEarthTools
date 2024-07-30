# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.
"""
Numpy Operations

| Category | Description | Available |
| -------- | ----------- | --------- |
| augument | Augument numpy data | `Rotate`, `Flip`, `Transform` | 
| conversion | Convert between data types | `ToXarray`, `ToDask` |
| filters | Filter data when iterating | `DropAnyNan`, `DropAllNan`, `DropValue`, `Shape` |
| join | Combine tuples of `np.ndarrays` | `Stack`, `VStack`, `HStack`, `Concatenate` |
| normalisation | Normalise arrays | `Anomaly`, `Deviation`, `Division`, `Evaluated`  |
| reshape | Reshape numpy array | `Rearrange`, `Squish`, `Expand`, `Flatten`, `SwapAxis` |
| select | Select elements from array | `Select`, `Slice` |
| split  | Split numpy arrays into tuples | `OnAxis`, `OnSlice`, `VSplit`, `HSplit` |
| values | Modify values of arrays | `FillNan`, `MaskValue`, `ForceNormalised` |
"""
from edit.pipeline.operations.numpy.join import Stack, Concatenate, VStack, HStack

from edit.pipeline.operations.numpy import (
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
    "augment",
    "filters",
    "reshape",
    "select",
    "split",
    "values",
    "normalisation",
    "conversion",
]
