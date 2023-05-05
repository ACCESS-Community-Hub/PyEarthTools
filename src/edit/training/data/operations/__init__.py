"""
Collection of [DataOperations][edit.training.data.templates.DataOperation] for use in Data Pipelines

| Name                | Description |
| ------------------- | ----------- |
| [reshape][edit.training.data.operations.reshape]                  | Alter the shape of numpy arrays [numpy.ndarray]  |
| [values]edit.training.data.operations.values]                     | Change values in the data; FillNa, Mask; ForceNormalised |
| [filters][edit.training.data.operations.filters]                  | Filter Data when iterating but not on retrieval |
| [sampler][edit.training.data.operations.sampler]                  | Change Sampling routine of data |
| [PatchingDataIndex][edit.training.data.operations.patch]          | Patch Data into small arrays |
| [TransformOperation][edit.training.data.operations.transforms]    | Apply [Transforms][edit.data.Transform] to data|
"""

from edit.training.data.operations import reshape, values, filters, sampler
from edit.training.data.operations.patch import PatchingDataIndex
from edit.training.data.operations.transforms import TransformOperation
