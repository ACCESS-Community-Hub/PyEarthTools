# Data API Index

This is the data package which forms a part of the [PyEarthTools package](https://github.com/ACCESS-Community-Hub/PyEarthTools). It contains code for fetching, loading, transforming and working with a wide variety of data sources. It has support for industry standard data sources of common interest, and also has code to aid users in managing their own data in their own projects.

Many research facilities have pre-existing data holdings on disk, and it is not necessary for users to fetch data. On the other hand, many users do need to fetch their own data for their projects. Both situations are catered for, but it's important to bear in mind that this package is catering to a broad and diverse set of user requirements.

The use of the data package within PyEarthTools includes:
  - Fetching known data sources (such as ERA5 or ISD)
  - Indexing into either pre-existing or newly-fetched data that has been downloaded
  - Subsetting and reprocessing that data for efficient storage, access and reprocessing
  - Loading that data into memory for efficient use in machine learning
  - Performing scientific operations on that data as part of data pre-processing

These tasks are aided by the API presented by the data package. Users looking for "how-to guides" or worked examples should review the [Tutorial Gallery](https://pyearthtools.readthedocs.io/en/latest/notebooks/Gallery.html).

The rest of this page contains reference information for the components of the Data package. The data API docs can be viewed at [Data API Docs](data_api.md).

|  Module          |               Purpose                         |   API Docs    |
|------------------|-----------------------------------------------|---------------------|
|  `data.archive`  | Indexing and loading from known data holdings | - [ZarrIndex](data_api.md#pyearthtools.data.archive.ZarrIndex)     |
|                  |                                               | - [ZarrTimeIndex](data_api.md#pyearthtools.data.archive.ZarrTimeIndex) |
|                  |                                               | - [extensions.register_archive](data_api.md#pyearthtools.data.archive.extensions.register_archive) |
|                  |                                               | - [reset_root](data_api.md#pyearthtools.data.archive.reset_root) |
|                  |                                               | - [set_root](data_api.md#pyearthtools.data.archive.set_root) |
|                  |                                               | - [config_root](data_api.md#pyearthtools.data.archive.config_root) |
| `data.derived`   |   Calculated derived fields                   | - [DerivedValue](data_api.md#pyearthtools.data.derived.DerivedValue) |
|                  |                                               | - [TimeDerivedValue](data_api.md#pyearthtools.data.derived.TimeDerivedValue) |
|                  |                                               | - [AdvancedTimeDerivedValue](data_api.md#pyearthtools.data.derived.AdvancedTimeDerivedValue) |
|                  |                                               | - [Insolation](data_api.md#pyearthtools.data.derived.Insolation) |
| `data.download`  | Publicly available datasets                   | - [ARCOERA5](data_api.md#pyearthtools.data.download.arcoera5.ARCOERA5) |
|                  |                                               | - [WB2ERA5](data_api.md#pyearthtools.data.download.weatherbench.WB2ERA5) |
|                  |                                               | - [WB2ERA5Clim](data_api.md#pyearthtools.data.download.weatherbench.WB2ERA5Clim) |
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
|                  |                                               | - [cacheIndex.BaseCacheIndex](data_api.md#pyearthtools.data.indexes.cacheIndex.BaseCacheIndex) |
|                  |                                               | - [cacheIndex.FileSystemCacheIndex](data_api.md#pyearthtools.data.indexes.cacheIndex.FileSystemCacheIndex) |
|                  |                                               | - [cacheIndex.CacheFactory](data_api.md#pyearthtools.data.indexes.cacheIndex.CacheFactory) |
|                  |                                               | - [cacheIndex.FunctionalCache](data_api.md#pyearthtools.data.indexes.cacheIndex.FunctionalCache) |
|                  |                                               | - [combine.InterpolationIndex](data_api.md#pyearthtools.data.indexes.combine.InterpolationIndex) |
|                  |                                               | - [fake.FakeIndex](data_api.md#pyearthtools.data.indexes.fake.FakeIndex) |
|                  |                                               | - [extensions.register_accessor](data_api.md#pyearthtools.data.indexes.extensions.register_accessor) |
| `data.modifications`   |                                         | - [Modification](data_api.md#pyearthtools.data.modifications.Modification) |
|                  |                                               | - [register_modification](data_api.md#pyearthtools.data.modifications.register_modification) |
|                  |                                               | - [variable_modifications](data_api.md#pyearthtools.data.modifications.variable_modifications) |
|                  |                                               | - [aggregations.Aggregation](data_api.md#pyearthtools.data.modifications.aggregations.Aggregation) |
|                  |                                               | - [aggregations.AggregationGeneral](data_api.md#pyearthtools.data.modifications.aggregations.AggregationGeneral) |
|                  |                                               | - [aggregations.Mean](data_api.md#pyearthtools.data.modifications.aggregations.Mean) |
|                  |                                               | - [aggregations.Accumulate](data_api.md#pyearthtools.data.modifications.aggregations.Accumulate) |
|                  |                                               | - [constants.Constant](data_api.md#pyearthtools.data.modifications.constants.Constant) |
|                  |                                               | - [decorator.VariableModification](data_api.md#pyearthtools.data.modifications.decorator.VariableModification) |
|                  |                                               | - [decorator.Modifier](data_api.md#pyearthtools.data.modifications.decorator.Modifier) |
|                  |                                               | - [reductions.Reduction](data_api.md#pyearthtools.data.modifications.reductions.Reduction) |
|                  |                                               | - [reductions.Groupby](data_api.md#pyearthtools.data.modifications.reductions.Groupby) |
|                  |                                               | - [reductions.Hourly](data_api.md#pyearthtools.data.modifications.reductions.Hourly) |
|                  |                                               | - [reductions.Daily](data_api.md#pyearthtools.data.modifications.reductions.Daily) |
|                  |                                               | - [reductions.Monthly](data_api.md#pyearthtools.data.modifications.reductions.Monthly) |
|                  |                                               | - [register.register_modification](data_api.md#pyearthtools.data.modifications.register.register_modification) |
| `data.operations`  |                                             | - [percentile](data_api.md#pyearthtools.data.operations.percentile) |
|                  |                                               | - [aggregation](data_api.md#pyearthtools.data.operations.aggregation) |
|                  |                                               | - [binning](data_api.md#pyearthtools.data.operations.binning) |
|                  |                                               | - [SpatialInterpolation](data_api.md#pyearthtools.data.operations.SpatialInterpolation) |
|                  |                                               | - [TemporalInterpolation](data_api.md#pyearthtools.data.operations.TemporalInterpolation) |
|                  |                                               | - [FullInterpolation](data_api.md#pyearthtools.data.operations.FullInterpolation) |
|                  |                                               | - [index_routines.series](data_api.md#pyearthtools.data.operations.index_routines.series) |
|                  |                                               | - [index_routines.safe_series](data_api.md#pyearthtools.data.operations.index_routines.safe_series) |
|                  |                                               | - [index_operations.split_ds](data_api.md#pyearthtools.data.operations.index_operations.split_ds) |
|                  |                                               | - [index_operations.split_ds_gen](data_api.md#pyearthtools.data.operations.index_operations.split_ds_gen) |
|                  |                                               | - [index_operations.aggregation](data_api.md#pyearthtools.data.operations.index_operations.aggregation) |
|                  |                                               | - [index_operations.find_range](data_api.md#pyearthtools.data.operations.index_operations.find_range) |
|                  |                                               | - [utils.identify_time_dimension](data_api.md#pyearthtools.data.operations.utils.identify_time_dimension) |
|                  |                                               | - [forecast_op.forecast_series](data_api.md#pyearthtools.data.operations.forecast_op.forecast_series) |
|                  |                                               | - [forecast_op.forecast_as_basetime](data_api.md#pyearthtools.data.operations.forecast_op.forecast_as_basetime) |
|                  |                                               | - [forecast_op.forecast_select_time](data_api.md#pyearthtools.data.operations.forecast_op.forecast_select_time) |
| `data.patterns`  |                                               | - [PatternIndex](data_api.md#pyearthtools.data.patterns.PatternIndex) |
|                  |                                               | - [PatternTimeIndex](data_api.md#pyearthtools.data.patterns.PatternTimeIndex) |
|                  |                                               | - [PatternForecastIndex](data_api.md#pyearthtools.data.patterns.PatternForecastIndex) |
|                  |                                               | - [PatternVariableAware](data_api.md#pyearthtools.data.patterns.PatternVariableAware) |
|                  |                                               | - [Argument](data_api.md#pyearthtools.data.patterns.Argument) |
|                  |                                               | - [ArgumentExpansion](data_api.md#pyearthtools.data.patterns.ArgumentExpansion) |
|                  |                                               | - [ArgumentExpansionVariable](data_api.md#pyearthtools.data.patterns.ArgumentExpansionVariable) |
|                  |                                               | - [Direct](data_api.md#pyearthtools.data.patterns.Direct) |
|                  |                                               | - [TemporalDirect](data_api.md#pyearthtools.data.patterns.TemporalDirect) |
|                  |                                               | - [ForecastDirect](data_api.md#pyearthtools.data.patterns.ForecastDirect) |
|                  |                                               | - [DirectVariable](data_api.md#pyearthtools.data.patterns.DirectVariable) |
|                  |                                               | - [ForecastDirectVariable](data_api.md#pyearthtools.data.patterns.ForecastDirectVariable) |
|                  |                                               | - [TemporalDirectVariable](data_api.md#pyearthtools.data.patterns.TemporalDirectVariable) |
|                  |                                               | - [DirectFactory](data_api.md#pyearthtools.data.patterns.DirectFactory) |
|                  |                                               | - [ExpandedDate](data_api.md#pyearthtools.data.patterns.ExpandedDate) |
|                  |                                               | - [TemporalExpandedDate](data_api.md#pyearthtools.data.patterns.TemporalExpandedDate) |
|                  |                                               | - [ForecastExpandedDate](data_api.md#pyearthtools.data.patterns.ForecastExpandedDate) |
|                  |                                               | - [ExpandedDateVariable](data_api.md#pyearthtools.data.patterns.ExpandedDateVariable) |
|                  |                                               | - [ForecastExpandedDateVariable](data_api.md#pyearthtools.data.patterns.ForecastExpandedDateVariable) |
|                  |                                               | - [TemporalExpandedDateVariable](data_api.md#pyearthtools.data.patterns.TemporalExpandedDateVariable) |
|                  |                                               | - [ExpandedDateFactory](data_api.md#pyearthtools.data.patterns.ExpandedDateFactory) |
|                  |                                               | - [Static](data_api.md#pyearthtools.data.patterns.Static) |
|                  |                                               | - [ParsingPattern](data_api.md#pyearthtools.data.patterns.ParsingPattern) |
|                  |                                               | - [ZarrIndex](data_api.md#pyearthtools.data.patterns.ZarrIndex) |
|                  |                                               | - [ZarrTimeIndex](data_api.md#pyearthtools.data.patterns.ZarrTimeIndex) |
| `data.save`      |                                               | - [save.save](data_api.md#pyearthtools.data.save.save) |
|                  |                                               | - [ManageFiles](data_api.md#pyearthtools.data.save.ManageFiles) |
|                  |                                               | - [ManageTemp](data_api.md#pyearthtools.data.save.ManageTemp) |
|                  |                                               | - [array.save](data_api.md#pyearthtools.data.save.array.save) |
|                  |                                               | - [dask.save](data_api.md#pyearthtools.data.save.dask.save) |
|                  |                                               | - [dataset.save](data_api.md#pyearthtools.data.save.dataset.save) |
|                  |                                               | - [dataset.to_netcdf](data_api.md#pyearthtools.data.save.dataset.to_netcdf) |
|                  |                                               | - [dataset.to_zarr](data_api.md#pyearthtools.data.save.dataset.to_zarr) |
|                  |                                               | - [jsonsave.save](data_api.md#pyearthtools.data.save.jsonsave.save) |
|                  |                                               | - [plot.save](data_api.md#pyearthtools.data.save.plot.save) |
|                  |                                               | - [save_utils.check_if_exists](data_api.md#pyearthtools.data.save.save_utils.check_if_exists) |
|                  |                                               | - [save_utils.make_new_filename](data_api.md#pyearthtools.data.save.save_utils.make_new_filename) |
|                  |                                               | - [save_utils.keep_clear](data_api.md#pyearthtools.data.save.save_utils.keep_clear) |
| `data.transforms`   |                                            | - [Transform](data_api.md#pyearthtools.data.transforms.Transform) |
|                  |                                               | - [TransformCollection](data_api.md#pyearthtools.data.transforms.TransformCollection) |
|                  |                                               | - [FunctionTransform](data_api.md#pyearthtools.data.transforms.FunctionTransform) |
|                  |                                               | - [Derive](data_api.md#pyearthtools.data.transforms.Derive) |
|                  |                                               | - [aggregation.over](data_api.md#pyearthtools.data.transforms.aggregation.over) |
|                  |                                               | - [aggregation.leaving](data_api.md#pyearthtools.data.transforms.aggregation.leaving) |
|                  |                                               | - [aggregation.Aggregate](data_api.md#pyearthtools.data.transforms.aggregation.Aggregate) |
|                  |                                               | - [attributes.SetAttributes](data_api.md#pyearthtools.data.transforms.attributes.SetAttributes) |
|                  |                                               | - [attributes.SetEncoding](data_api.md#pyearthtools.data.transforms.attributes.SetEncoding) |
|                  |                                               | - [attributes.SetType](data_api.md#pyearthtools.data.transforms.attributes.SetType) |
|                  |                                               | - [attributes.Rename](data_api.md#pyearthtools.data.transforms.attributes.Rename) |
|                  |                                               | - [coordinates.get_longitude](data_api.md#pyearthtools.data.transforms.coordinates.get_longitude) |
|                  |                                               | - [coordinates.weak_cast_to_int](data_api.md#pyearthtools.data.transforms.coordinates.weak_cast_to_int) |
|                  |                                               | - [coordinates.StandardLongitude](data_api.md#pyearthtools.data.transforms.coordinates.StandardLongitude) |
|                  |                                               | - [coordinates.ReIndex](data_api.md#pyearthtools.data.transforms.coordinates.ReIndex) |
|                  |                                               | - [coordinates.StandardCoordinateNames](data_api.md#pyearthtools.data.transforms.coordinates.StandardCoordinateNames) |
|                  |                                               | - [coordinates.Select](data_api.md#pyearthtools.data.transforms.coordinates.Select) |
|                  |                                               | - [coordinates.Drop](data_api.md#pyearthtools.data.transforms.coordinates.Drop) |
|                  |                                               | - [coordinates.Flatten](data_api.md#pyearthtools.data.transforms.coordinates.Flatten) |
|                  |                                               | - [coordinates.Expand](data_api.md#pyearthtools.data.transforms.coordinates.Expand) |
|                  |                                               | - [coordinates.SelectFlatten](data_api.md#pyearthtools.data.transforms.coordinates.SelectFlatten) |
|                  |                                               | - [coordinates.Assign](data_api.md#pyearthtools.data.transforms.coordinates.Assign) |
|                  |                                               | - [coordinates.Pad](data_api.md#pyearthtools.data.transforms.coordinates.Pad) |
|                  |                                               | - [default.get_default_transforms](data_api.md#pyearthtools.data.transforms.default.get_default_transforms) |
|                  |                                               | - [derive.evaluate](data_api.md#pyearthtools.data.transforms.derive.evaluate) |
|                  |                                               | - [derive.derive_equations](data_api.md#pyearthtools.data.transforms.derive.derive_equations) |
|                  |                                               | - [dimensions.StandardDimensionNames](data_api.md#pyearthtools.data.transforms.dimensions.StandardDimensionNames) |
|                  |                                               | - [dimensions.Expand](data_api.md#pyearthtools.data.transforms.dimensions.Expand) |
|                  |                                               | - [interpolation.Interpolate](data_api.md#pyearthtools.data.transforms.interpolation.Interpolate) |
|                  |                                               | - [interpolation.XESMF](data_api.md#pyearthtools.data.transforms.interpolation.XESMF) |
|                  |                                               | - [interpolation.InterpolateNan](data_api.md#pyearthtools.data.transforms.interpolation.InterpolateNan) |
|                  |                                               | - [interpolation.like](data_api.md#pyearthtools.data.transforms.interpolation.like) |
|                  |                                               | - [mask.UnderlyingMaskTransform](data_api.md#pyearthtools.data.transforms.mask.UnderlyingMaskTransform) |
|                  |                                               | - [mask.Dataset](data_api.md#pyearthtools.data.transforms.mask.Dataset) |
|                  |                                               | - [mask.Replace](data_api.md#pyearthtools.data.transforms.mask.Replace) |
|                  |                                               | - [optimisation.Rechunk](data_api.md#pyearthtools.data.transforms.optimisation.Rechunk) |
|                  |                                               | - [region.check_shape](data_api.md#pyearthtools.data.transforms.region.check_shape) |
|                  |                                               | - [region.order](data_api.md#pyearthtools.data.transforms.region.order) |
|                  |                                               | - [region.like](data_api.md#pyearthtools.data.transforms.region.like) |
|                  |                                               | - [region.Bounding](data_api.md#pyearthtools.data.transforms.region.Bounding) |
|                  |                                               | - [region.Select](data_api.md#pyearthtools.data.transforms.region.Select) |
|                  |                                               | - [region.ISelect](data_api.md#pyearthtools.data.transforms.region.ISelect) |
|                  |                                               | - [region.PointBox](data_api.md#pyearthtools.data.transforms.region.PointBox) |
|                  |                                               | - [region.Lookup](data_api.md#pyearthtools.data.transforms.region.Lookup) |
|                  |                                               | - [region.Geosearch](data_api.md#pyearthtools.data.transforms.region.Geosearch) |
|                  |                                               | - [region.ShapeFile](data_api.md#pyearthtools.data.transforms.region.ShapeFile) |
|                  |                                               | - [utils.parse_dataset](data_api.md#pyearthtools.data.transforms.utils.parse_dataset) |
| `data.catalog`   |                                               | - [Catalog](data_api.md#pyearthtools.data.catalog.Catalog) |
|                  |                                               | - [CatalogEntry](data_api.md#pyearthtools.data.catalog.CatalogEntry) |
|                  |                                               | - [get_name](data_api.md#pyearthtools.data.catalog.get_name) |
| `data.collection`   |                                            | - [Collection](data_api.md#pyearthtools.data.collection.Collection) |
|                  |                                               | - [LabelledCollection](data_api.md#pyearthtools.data.collection.LabelledCollection) |
| `data.exceptions`   |                                            | - [InvalidIndexError](data_api.md#pyearthtools.data.exceptions.InvalidIndexError) |
|                  |                                               | - [InvalidDataError](data_api.md#pyearthtools.data.exceptions.InvalidDataError) |
|                  |                                               | - [DataNotFoundError](data_api.md#pyearthtools.data.exceptions.DataNotFoundError) |
| `data.load`      |                                               | - [load](data_api.md#pyearthtools.data.load.load) |
| `data.time`      |                                               | - [multisplit](data_api.md#pyearthtools.data.time.multisplit) |
|                  |                                               | - [find_components](data_api.md#pyearthtools.data.time.find_components) |
|                  |                                               | - [strip_to_common_resolution](data_api.md#pyearthtools.data.time.strip_to_common_resolution) |
|                  |                                               | - [time_delta](data_api.md#pyearthtools.data.time.time_delta) |
|                  |                                               | - [time_delta_resolution](data_api.md#pyearthtools.data.time.time_delta_resolution) |
|                  |                                               | - [range_samples](data_api.md#pyearthtools.data.time.range_samples) |
|                  |                                               | - [TimeResolution](data_api.md#pyearthtools.data.time.TimeResolution) |
|                  |                                               | - [Petdt](data_api.md#pyearthtools.data.time.Petdt) |
|                  |                                               | - [TimeDelta](data_api.md#pyearthtools.data.time.TimeDelta) |
|                  |                                               | - [TimeRange](data_api.md#pyearthtools.data.time.TimeRange) |
| `data.warnings`  |                                               | - [pyearthtoolsDataWarning](data_api.md#pyearthtools.data.warnings.InvalidIndexError) |
|                  |                                               | - [IndexWarning](data_api.md#pyearthtools.data.warnings.IndexWarning) |
|                  |                                               | - [AccessorRegistrationWarning](data_api.md#pyearthtools.data.warnings.AccessorRegistrationWarning) |


