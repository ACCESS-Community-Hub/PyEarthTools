# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

# ruff: noqa: F401

"""
Indexes for EDIT.

These are the effective backbone of all of `edit`, providing the API in which to retrieve data.
Shown below are all `indexes` which are used by both the [archive][edit.data.archive], [pattern][edit.data.patterns],
and the [static][edit.data.static] data sources.

## Indexes
| Index | Purpose |
| ----- | ------------------------ |
| [Index][edit.data.indexes.Index] | Base Index to define API and common functions |
| [FileSystemIndex][edit.data.indexes.FileSystemIndex] | Add filesystem retrieval |
| [DataIndex][edit.data.indexes.DataIndex] | Introduce Transforms |
| [TimeIndex][edit.data.indexes.TimeIndex] | Add time specific indexing |
| [AdvancedTimeIndex][edit.data.indexes.AdvancedTimeIndex] | Extend time indexing for advanced uses.|
| [AdvancedTimeDataIndex][edit.data.indexes.AdvancedTimeDataIndex] | Combine AdvancedTimeIndex and DataIndex |
| [ArchiveIndex][edit.data.indexes.ArchiveIndex] | Default class for Archived data |
| [ForecastIndex][edit.data.indexes.ForecastIndex] | Base class for forecast data, combines DataIndex and FileSystemIndex |
| [StaticDataIndex][edit.data.indexes.StaticDataIndex] | Base class for static on disk data, combines DataIndex and FileSystemIndex |
| [CachingIndex][edit.data.indexes.CachingIndex] | Data generated on the fly cached to a given location |

## Usage
To use the indexes, or to extend EDIT's capability to a new dataset or data source, one of the above listed classes should be subclassed.

Which one depends on the use case and data specifications, but `ArchiveIndex`, `ForecastIndex` or `StaticDataIndex` are good places to start for on disk data,
with `DataIndex` or `CachingIndex` useful for ondemand generated data.

See [archive][edit.data.archive] for prebuilt indexes.

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

from edit.data.indexes.indexes import (
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
from edit.data.indexes.cacheIndex import (
    BaseCacheIndex,
    CachingIndex,
    CachingForecastIndex,
    FunctionalCacheIndex,
)
from edit.data.indexes import utilities, decorators
from edit.data.indexes.extensions import register_accessor

from edit.data.indexes.utilities.spellcheck import VariableDefault, VARIABLE_DEFAULT
from edit.data.indexes.utilities.structure import structure

from edit.data.indexes.decorators import alias_arguments, check_arguments

from edit.data.indexes.intake import IntakeIndex, IntakeIndexCache
from edit.data.indexes.templates import Structured

from edit.data.indexes.fake import FakeIndex

from edit.data.indexes.utilities.folder_size import ByteSize

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
