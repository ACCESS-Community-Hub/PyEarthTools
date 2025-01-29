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


# ruff: noqa: F401

"""
Indexes for pyearthtools.

These are the effective backbone of all of `pyearthtools`, providing the API in which to retrieve data.
Shown below are all `indexes` which are used by both the [archive][pyearthtools.data.archive], [pattern][pyearthtools.data.patterns],
and the [static][pyearthtools.data.static] data sources.

## Indexes
| Index | Purpose |
| ----- | ------------------------ |
| [Index][pyearthtools.data.indexes.Index] | Base Index to define API and common functions |
| [FileSystemIndex][pyearthtools.data.indexes.FileSystemIndex] | Add filesystem retrieval |
| [DataIndex][pyearthtools.data.indexes.DataIndex] | Introduce Transforms |
| [TimeIndex][pyearthtools.data.indexes.TimeIndex] | Add time specific indexing |
| [AdvancedTimeIndex][pyearthtools.data.indexes.AdvancedTimeIndex] | Extend time indexing for advanced uses.|
| [AdvancedTimeDataIndex][pyearthtools.data.indexes.AdvancedTimeDataIndex] | Combine AdvancedTimeIndex and DataIndex |
| [ArchiveIndex][pyearthtools.data.indexes.ArchiveIndex] | Default class for Archived data |
| [ForecastIndex][pyearthtools.data.indexes.ForecastIndex] | Base class for forecast data, combines DataIndex and FileSystemIndex |
| [StaticDataIndex][pyearthtools.data.indexes.StaticDataIndex] | Base class for static on disk data, combines DataIndex and FileSystemIndex |
| [CachingIndex][pyearthtools.data.indexes.CachingIndex] | Data generated on the fly cached to a given location |

## Usage
To use the indexes, or to extend pyearthtools's capability to a new dataset or data source, one of the above listed classes should be subclassed.

Which one depends on the use case and data specifications, but `ArchiveIndex`, `ForecastIndex` or `StaticDataIndex` are good places to start for on disk data,
with `DataIndex` or `CachingIndex` useful for ondemand generated data.

See [archive][pyearthtools.data.archive] for prebuilt indexes.

## Class Diagram

```mermaid
classDiagram
    Index <|-- FileSystemIndex
    Index <|-- DataIndex
    Index <|-- TimeIndex
    TimeIndex <|-- AdvancedTimeIndex
    DataIndex <| -- AdvancedTimeDataIndex
    AdvancedTimeIndex <| -- AdvancedTimeDataIndex
    FileSystemIndex <| -- ArchiveIndex
    AdvancedTimeDataIndex <| -- ArchiveIndex
    TimeIndex <| -- BaseTimeIndex
    DataFileSystemIndex <| -- BaseTimeIndex
    FileSystemIndex <|-- DataFileSystemIndex
    DataIndex <| -- DataFileSystemIndex
    DataFileSystemIndex <| -- StaticDataIndex
    DataFileSystemIndex <| -- ForecastIndex
    TimeIndex <| -- ForecastIndex

    class Index{
        Base Level Index
        +record_initialisation()
    }
    class FileSystemIndex{
        Allow Filesystem searching
      +dict ROOT_DIRECTORIES
      +search()
      +get()
    }
    class DataIndex{
        Add Transforms
      + Transform base_transforms
      +retrieve()
    }
    class TimeIndex{
        Basic Time based indexing
      +retrieve()
    }
    class AdvancedTimeIndex{
        Advanced Time based indexing
      +retrieve()
      +series()
      +safe_series()
      +aggregation()
      +range()
    }
    class AdvancedTimeDataIndex{
        Advanced Time and Transforms
    }
    class DataFileSystemIndex{
        Transforms and File System
    }
    class BaseTimeIndex{
        Transforms, File System and simple Time
    }
    class ArchiveIndex{
        Default class for Archives
        Is Transforms, FileSystem and AdvancedTime
    }
    class ForecastIndex{
        Forecast Data
    }
    class StaticDataIndex{
        Static Data
    }
```

"""

from pyearthtools.data.indexes.indexes import (
    Index,
    DataIndex,
    FileSystemIndex,
    TimeIndex,
    TimeDataIndex,
    AdvancedTimeIndex,
    AdvancedTimeDataIndex,
    BaseTimeIndex,
    DataFileSystemIndex,
    ArchiveIndex,
    ForecastIndex,
    StaticDataIndex,
)
from pyearthtools.data.indexes.cacheIndex import (
    FileSystemCacheIndex,
    CachingIndex,
    CachingForecastIndex,
    FunctionalCacheIndex,
    MemCache,
    FunctionalMemCacheIndex,
)
from pyearthtools.data.indexes import utilities, decorators
from pyearthtools.data.indexes.extensions import register_accessor

from pyearthtools.data.indexes.utilities.spellcheck import VariableDefault, VARIABLE_DEFAULT
from pyearthtools.data.indexes.utilities.structure import structure

from pyearthtools.data.indexes.decorators import alias_arguments, check_arguments

from pyearthtools.data.indexes.intake import IntakeIndex, IntakeIndexCache
from pyearthtools.data.indexes.templates import Structured

from pyearthtools.data.indexes.fake import FakeIndex

from pyearthtools.data.indexes.utilities.folder_size import ByteSize

__all__ = [
    "Index",
    "DataIndex",
    "FileSystemIndex",
    "TimeIndex",
    "TimeDataIndex",
    "AdvancedTimeIndex",
    "AdvancedTimeDataIndex",
    "BaseTimeIndex",
    "DataFileSystemIndex",
    "ArchiveIndex",
    "ForecastIndex",
    "StaticDataIndex",
    "CachingIndex",
    "CachingForecastIndex",
    "IntakeIndex",
    "IntakeIndexCache",
]
