# PyEarthTools Data Package

This is the data package which forms a part of the [PyEarthTools package](https://github.com/ACCESS-Community-Hub/PyEarthTools). It contains code for fetching, loading, transforming and working with a wide variety of data sources. It has support for industry standard data sources of common interest, and also has code to aid users in managing their own data in their own projects.

Many research facilities have pre-existing data holdings on disk, and it is not necessary for users to fetch data. On the other hand, many users do need to fetch their own data for their projects. Both situations are catered for, but it's important to bear in mind that this package is catering to a broad and diverse set of user requirements.

The use of the data package within PyEarthTools includes:
  - Fetching known data sources (such as ERA5 or ISD)
  - Indexing into either pre-existing or newly-fetched data that has been downloaded
  - Subsetting and reprocessing that data for efficient storage, access and reprocessing
  - Loading that data into memory for efficient use in machine learning
  - Performing scientific operations on that data as part of data pre-processing

These tasks are aided by the API presented by the data package. Users looking for "how-to guides" or worked examples should review the [Tutorial Gallery](https://pyearthtools.readthedocs.io/en/latest/notebooks/Gallery.html). 

The rest of this page contains reference information for the components of the Data package. The entire data API docs can be viewed at [Data API](data_api.md)

|  Module          |               Purpose                         |   API Docs    |
|------------------|-----------------------------------------------|---------------------|
|  `data.archive`  | Indexing and loading from known data holdings | - [ZarrIndex](data_api.md#pyearthtools.data.archive.ZarrIndex)     |
|                  |                                               | - [ZarrTimeIndex](data_api.md#pyearthtools.data.archive.ZarrTimeIndex) |
| `data.derived`   |   Calculated derived fields                   | - [DerivedValue](data_api.md#pyearthtools.data.derived.DerivedValue) |
|                  |                                               | - [TimeDerivedValue](data_api.md#pyearthtools.data.derived.TimeDerivedValue) |
|                  |                                               | - [AdvancedTimeDerivedValue](data_api.md#pyearthtools.data.derived.AdvancedTimeDerivedValue) |
|                  |                                               | - [Insolation](data_api.md#pyearthtools.data.derived.Insolation) |
| `data.indexes`   |                                               | - [Index](data_api.md#pyearthtools.data.indexes.Index) |
|                  |                                               | - [DataIndex](data_api.md#pyearthtools.data.indexes.DataIndex) |
|                  |                                               | - [FileSystemIndex](data_api.md#pyearthtools.data.indexes.FileSystemIndex) |
|                  |                                               | - [TimeIndex](data_api.md#pyearthtools.data.indexes.TimeIndex) |
|                  |                                               | - [TimeDataIndex](data_api.md#pyearthtools.data.indexes.TimeDataIndex) |
|                  |                                               | - [AdvancedTimeIndex](data_api.md#pyearthtools.data.indexes.AdvancedTimeIndex) |
|                  |                                               | - [AdvancedTimeDataIndex](data_api.md#pyearthtools.data.indexes.AdvancedTimeDataIndex) |
|                  |                                               | - [BaseTimeIndex](data_api.md#pyearthtools.data.indexes.BaseTimeIndex) |
|                  |                                               | - [DataFileSystemIndex](data_api.md#pyearthtools.data.indexes.DataFileSystemIndex) |
|                  |                                               | - [ArchiveIndex](data_api.md#pyearthtools.data.indexes.ArchiveIndex) |
|                  |                                               | - [ForecastIndex](data_api.md#pyearthtools.data.indexes.ForecastIndex) |
|                  |                                               | - [StaticDataIndex](data_api.md#pyearthtools.data.indexes.StaticDataIndex) |
|                  |                                               | - [CachingForecastIndex](data_api.md#pyearthtools.data.indexes.CachingForecastIndex) |
|                  |                                               | - [IntakeIndex](data_api.md#pyearthtools.data.indexes.IntakeIndex) |
|                  |                                               | - [IntakeIndexCache](data_api.md#pyearthtools.data.indexes.IntakeIndexCache) |

    

