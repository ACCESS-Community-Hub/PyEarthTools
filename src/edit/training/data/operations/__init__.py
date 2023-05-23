"""
Collection of [DataOperations][edit.training.data.templates.DataOperation] for use in Data Pipelines

| Name                | Description |
| ------------------- | ----------- |
| [reshape][edit.training.data.operations.reshape]                  | Alter the shape of numpy arrays [numpy.ndarray]; Reshape, Squish, Expand, Flatten  |
| [values][edit.training.data.operations.values]                    | Change values in the data; FillNa, Mask, ForceNormalised |
| [filters][edit.training.data.operations.filters]                  | Filter Data when iterating but not on retrieval; DropNan, DropValue |
| [sampler][edit.training.data.operations.sampler]                  | Change Sampling routine of data; RandomSample, RandomDropOut, DropOut Interval,  |
| [sort][edit.training.data.operations.sort]                        | Sort Data Variables  |
| [pad][edit.training.data.operations.pad]                          | Pad Data  |
| [PatchingDataIndex][edit.training.data.operations.patch]          | Patch Data into small arrays |
| [ToNumpy][edit.training.data.operations.to_numpy]                 | Convert xarray's to numpy's |
| [TransformOperation][edit.training.data.operations.transforms]    | Apply [Transforms][edit.data.Transform] to data|
"""

from edit.training.data.operations import reshape, values, filters, sampler, sort
from edit.training.data.operations.patch import PatchingDataIndex
from edit.training.data.operations.to_numpy import ToNumpy
from edit.training.data.operations.transforms import TransformOperation
from edit.training.data.operations.select import Select

from edit.training.data.operations.filters import DataFilter
from edit.training.data.operations.sampler import DataSampler
