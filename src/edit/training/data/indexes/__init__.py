"""
edit.training purpose built [DataIndexes][edit.data.DataIndex].

| Name                | Description |
| ------------------- | ----------- |
| [InterpolationIndex][edit.training.data.indexes.interpolate]  | Interpolate DataIndexes onto the same Spatial Grid and/or Temporal Resolution |
| [CoordinateIndex][edit.training.data.indexes.coordinate]     | Add Coordinates as Data Variables to the Datasets |
| [CachingIndex][edit.training.data.indexes.cache]        | Cache Datasets out to directory |
| [TemporalIndex][edit.training.data.indexes.temporal]       | Add Time dimension to data |

"""

from edit.training.data.indexes.interpolate import InterpolationIndex
from edit.training.data.indexes.coordinate import CoordinateIndex
from edit.training.data.indexes.cache import CachingIndex
from edit.training.data.indexes.temporal import TemporalIndex
