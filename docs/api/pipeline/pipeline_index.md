# Pipeline API Index

This is the Pipeline package which forms a part of the [PyEarthTools package](https://github.com/ACCESS-Community-Hub/PyEarthTools).

The rest of this page contains reference information for the components of the Pipeline package. The Pipeline API docs can be viewed at [Pipeline API Docs](pipeline_api.md).

|  Module              |       Purpose                        |   API Docs     |
|----------------------|--------------------------------------|----------------|
|  `pipeline`          |                                      | - [Sampler](pipeline_api.md#pyearthtools.pipeline.Sampler)  |
|                      |                                      | - [Iterator](pipeline_api.md#pyearthtools.pipeline.Iterator)  |
|                      |                                      | - [Pipeline](pipeline_api.md#pyearthtools.pipeline.Pipeline)  |
|                      |                                      | - [Operation](pipeline_api.md#pyearthtools.pipeline.Operation)  |
|                      |                                      | - [PipelineException](pipeline_api.md#pyearthtools.pipeline.PipelineException)  |
|                      |                                      | - [PipelineFilterException](pipeline_api.md#pyearthtools.pipeline.PipelineFilterException)  |
|                      |                                      | - [PipelineRuntimeError](pipeline_api.md#pyearthtools.pipeline.PipelineRuntimeError)  |
|                      |                                      | - [PipelineTypeError](pipeline_api.md#pyearthtools.pipeline.PipelineTypeError)  |
| `pipeline.branching` |                                      | - [PipelineBranchPoint](pipeline_api.md#pyearthtools.pipeline.branching.PipelineBranchPoint)  |
|                      |                                      | - [Unifier](pipeline_api.md#pyearthtools.pipeline.branching.Unifier)  |
|                      |                                      | - [Joiner](pipeline_api.md#pyearthtools.pipeline.branching.Joiner)  |
|                      |                                      | - [Spliter](pipeline_api.md#pyearthtools.pipeline.branching.Spliter)  |
| `pipeline.filters`   |                                      | - [Filter](pipeline_api.md#pyearthtools.pipeline.filters.Filter)  |
|                      |                                      | - [FilterCheck](pipeline_api.md#pyearthtools.pipeline.filters.FilterCheck)  |
|                      |                                      | - [FilterWarningContext](pipeline_api.md#pyearthtools.pipeline.filters.FilterWarningContext)  |
|                      |                                      | - [TypeFilter](pipeline_api.md#pyearthtools.pipeline.filters.TypeFilter)  |
| `pipeline.iterators` |                                      | - [Iterator](pipeline_api.md#pyearthtools.pipeline.iterators.Iterator)  |
|                      |                                      | - [Range](pipeline_api.md#pyearthtools.pipeline.iterators.Range)  |
|                      |                                      | - [Predefined](pipeline_api.md#pyearthtools.pipeline.iterators.Predefined)  |
|                      |                                      | - [File](pipeline_api.md#pyearthtools.pipeline.iterators.File)  |
|                      |                                      | - [DateRange](pipeline_api.md#pyearthtools.pipeline.iterators.DateRange)  |
|                      |                                      | - [DateRangeLimit](pipeline_api.md#pyearthtools.pipeline.iterators.DateRangeLimit)  |
|                      |                                      | - [Randomise](pipeline_api.md#pyearthtools.pipeline.iterators.Randomise)  |
|                      |                                      | - [SuperIterator](pipeline_api.md#pyearthtools.pipeline.iterators.SuperIterator)  |
|                      |                                      | - [IterateResults](pipeline_api.md#pyearthtools.pipeline.iterators.IterateResults)  |
| `pipeline.modifications` |                                  | - [Cache](pipeline_api.md#pyearthtools.pipeline.modifications.Cache)  |
|                      |                                      | - [StaticCache](pipeline_api.md#pyearthtools.pipeline.modifications.StaticCache)  |
|                      |                                      | - [MemCache](pipeline_api.md#pyearthtools.pipeline.modifications.MemCache)  |
|                      |                                      | - [IdxModifier](pipeline_api.md#pyearthtools.pipeline.modifications.IdxModifier)  |
|                      |                                      | - [IdxOverride](pipeline_api.md#pyearthtools.pipeline.modifications.IdxOverride)  |
|                      |                                      | - [TimeIdxModifier](pipeline_api.md#pyearthtools.pipeline.modifications.TimeIdxModifier)  |
|                      |                                      | - [SequenceRetrieval](pipeline_api.md#pyearthtools.pipeline.modifications.SequenceRetrieval)  |
|                      |                                      | - [TemporalRetrieval](pipeline_api.md#pyearthtools.pipeline.modifications.TemporalRetrieval)  |
|                      |                                      | - [idx_modification](pipeline_api.md#pyearthtools.pipeline.modifications.idx_modification)  |
| `pipeline.operations`  |                                    | - [Transforms](pipeline_api.md#pyearthtools.pipeline.operations.Transforms)  |
| `pipeline.operations.xarray`  |                             | - [Compute](pipeline_api.md#pyearthtools.pipeline.operations.xarray.Compute)  |
|                      |                                      | - [Merge](pipeline_api.md#pyearthtools.pipeline.operations.xarray.Merge)  |
|                      |                                      | - [Concatenate](pipeline_api.md#pyearthtools.pipeline.operations.xarray.Concatenate)  |
|                      |                                      | - [Sort](pipeline_api.md#pyearthtools.pipeline.operations.xarray.Sort)  |
|                      |                                      | - [Chunk](pipeline_api.md#pyearthtools.pipeline.operations.xarray.Chunk)  |
|                      |                                      | - [RecodeCalendar](pipeline_api.md#pyearthtools.pipeline.operations.xarray.RecodeCalendar)  |
|                      |                                      | - [AlignDates](pipeline_api.md#pyearthtools.pipeline.operations.xarray.AlignDates)  |
| `pipeline.operations.xarray.conversion`  |                  | - [ToNumpy](pipeline_api.md#pyearthtools.pipeline.operations.xarray.conversion.ToNumpy)  |
|                      |                                      | - [ToDask](pipeline_api.md#pyearthtools.pipeline.operations.xarray.conversion.ToDask)  |
| `pipeline.operations.xarray.filters`  |                     | - [XarrayFilter](pipeline_api.md#pyearthtools.pipeline.operations.xarray.filters.XarrayFilter)  |
|                      |                                      | - [DropAnyNan](pipeline_api.md#pyearthtools.pipeline.operations.xarray.filters.DropAnyNan)  |
|                      |                                      | - [DropAllNan](pipeline_api.md#pyearthtools.pipeline.operations.xarray.filters.DropAllNan)  |
|                      |                                      | - [DropValue](pipeline_api.md#pyearthtools.pipeline.operations.xarray.filters.DropValue)  |
|                      |                                      | - [Shape](pipeline_api.md#pyearthtools.pipeline.operations.xarray.filters.Shape)  |
| `pipeline.operations.xarray.reshape`  |                     | - [Dimensions](pipeline_api.md#pyearthtools.pipeline.operations.xarray.reshape.Dimensions)  |
|                      |                                      | - [CoordinateFlatten](pipeline_api.md#pyearthtools.pipeline.operations.xarray.reshape.CoordinateFlatten)  |
| `pipeline.operations.xarray.select`  |                      | - [SelectDataset](pipeline_api.md#pyearthtools.pipeline.operations.xarray.select.SelectDataset)  |
|                      |                                      | - [DropDataset](pipeline_api.md#pyearthtools.pipeline.operations.xarray.select.DropDataset)  |
|                      |                                      | - [SliceDataset](pipeline_api.md#pyearthtools.pipeline.operations.xarray.select.SliceDataset)  |
| `pipeline.operations.xarray.split`  |                       | - [OnVariables](pipeline_api.md#pyearthtools.pipeline.operations.xarray.split.OnVariables)  |
|                      |                                      | - [OnCoordinate](pipeline_api.md#pyearthtools.pipeline.operations.xarray.split.OnCoordinate)  |
| `pipeline.operations.xarray.values`  |                      | - [FillNan](pipeline_api.md#pyearthtools.pipeline.operations.xarray.values.FillNan)  |
|                      |                                      | - [MaskValue](pipeline_api.md#pyearthtools.pipeline.operations.xarray.values.MaskValue)  |
|                      |                                      | - [ForceNormalised](pipeline_api.md#pyearthtools.pipeline.operations.xarray.values.ForceNormalised)  |
|                      |                                      | - [Derive](pipeline_api.md#pyearthtools.pipeline.operations.xarray.values.Derive)  |
| `pipeline.operations.xarray.metadata`  |                    | - [Rename](pipeline_api.md#pyearthtools.pipeline.operations.xarray.metadata.Rename)  |
|                      |                                      | - [Encoding](pipeline_api.md#pyearthtools.pipeline.operations.xarray.metadata.Encoding)  |
|                      |                                      | - [MaintainEncoding](pipeline_api.md#pyearthtools.pipeline.operations.xarray.metadata.MaintainEncoding)  |
|                      |                                      | - [Attributes](pipeline_api.md#pyearthtools.pipeline.operations.xarray.metadata.Attributes)  |
|                      |                                      | - [MaintainAttributes](pipeline_api.md#pyearthtools.pipeline.operations.xarray.metadata.MaintainAttributes)  |
| `pipeline.operations.xarray.normalisation`  |               | - [xarrayNormalisation](pipeline_api.md#pyearthtools.pipeline.operations.xarray.normalisation.xarrayNormalisation)  |
|                      |                                      | - [Anomaly](pipeline_api.md#pyearthtools.pipeline.operations.xarray.normalisation.Anomaly)  |
|                      |                                      | - [Deviation](pipeline_api.md#pyearthtools.pipeline.operations.xarray.normalisation.Deviation)  |
|                      |                                      | - [Division](pipeline_api.md#pyearthtools.pipeline.operations.xarray.normalisation.Division)  |
|                      |                                      | - [Evaluated](pipeline_api.md#pyearthtools.pipeline.operations.xarray.normalisation.Evaluated)  |
| `pipeline.operations.xarray.remapping`  |                   | - [HEALPix](pipeline_api.md#pyearthtools.pipeline.operations.xarray.remapping.HEALPix)  |
| `pipeline.samplers`  |                                      | - [EmptyObject](pipeline_api.md#pyearthtools.pipeline.samplers.EmptyObject)  |
|                      |                                      | - [Sampler](pipeline_api.md#pyearthtools.pipeline.samplers.Sampler)  |
|                      |                                      | - [Default](pipeline_api.md#pyearthtools.pipeline.samplers.Default)  |
|                      |                                      | - [SuperSampler](pipeline_api.md#pyearthtools.pipeline.samplers.SuperSampler)  |
|                      |                                      | - [Random](pipeline_api.md#pyearthtools.pipeline.samplers.Random)  |
|                      |                                      | - [DropOut](pipeline_api.md#pyearthtools.pipeline.samplers.DropOut)  |
|                      |                                      | - [RandomDropOut](pipeline_api.md#pyearthtools.pipeline.samplers.RandomDropOut)  |
| `pipeline.operations.dask`  |                               | - [Stack](pipeline_api.md#pyearthtools.pipeline.operations.dask.Stack)  |
|                      |                                      | - [Concatenate](pipeline_api.md#pyearthtools.pipeline.operations.dask.Concatenate)  |
|                      |                                      | - [VStack](pipeline_api.md#pyearthtools.pipeline.operations.dask.VStack)  |
|                      |                                      | - [HStack](pipeline_api.md#pyearthtools.pipeline.operations.dask.HStack)  |
|                      |                                      | - [Compute](pipeline_api.md#pyearthtools.pipeline.operations.dask.Compute)  |
| `pipeline.operations.dask.augment`  |                       | - [Rotate](pipeline_api.md#pyearthtools.pipeline.operations.dask.augment.Rotate)  |
|                      |                                      | - [Flip](pipeline_api.md#pyearthtools.pipeline.operations.dask.augment.Flip)  |
|                      |                                      | - [Transform](pipeline_api.md#pyearthtools.pipeline.operations.dask.augment.Transform)  |
| `pipeline.operations.dask.filters`  |                       | - [daskFilter](pipeline_api.md#pyearthtools.pipeline.operations.dask.filters.daskFilter)  |
|                      |                                      | - [DropAnyNan](pipeline_api.md#pyearthtools.pipeline.operations.dask.filters.DropAnyNan)  |
|                      |                                      | - [DropAllNan](pipeline_api.md#pyearthtools.pipeline.operations.dask.filters.DropAllNan)  |
|                      |                                      | - [DropValue](pipeline_api.md#pyearthtools.pipeline.operations.dask.filters.DropValue)  |
|                      |                                      | - [Shape](pipeline_api.md#pyearthtools.pipeline.operations.dask.filters.Shape)  |
| `pipeline.operations.dask.reshape`  |                       | - [Rearrange](pipeline_api.md#pyearthtools.pipeline.operations.dask.reshape.Rearrange)  |
|                      |                                      | - [Squeeze](pipeline_api.md#pyearthtools.pipeline.operations.dask.reshape.Squeeze)  |
|                      |                                      | - [Flattener](pipeline_api.md#pyearthtools.pipeline.operations.dask.reshape.Flattener)  |
|                      |                                      | - [Flatten](pipeline_api.md#pyearthtools.pipeline.operations.dask.reshape.Flatten)  |
|                      |                                      | - [SwapAxis](pipeline_api.md#pyearthtools.pipeline.operations.dask.reshape.SwapAxis)  |
| `pipeline.operations.dask.select`  |                        | - [Select](pipeline_api.md#pyearthtools.pipeline.operations.dask.select.Select)  |
|                      |                                      | - [Slice](pipeline_api.md#pyearthtools.pipeline.operations.dask.select.Slice)  |
| `pipeline.operations.dask.split`  |                         | - [OnAxis](pipeline_api.md#pyearthtools.pipeline.operations.dask.split.OnAxis)  |
|                      |                                      | - [OnSlice](pipeline_api.md#pyearthtools.pipeline.operations.dask.split.OnSlice)  |
| `pipeline.operations.dask.values`  |                        | - [FillNan](pipeline_api.md#pyearthtools.pipeline.operations.dask.values.FillNan)  |
|                      |                                      | - [MaskValue](pipeline_api.md#pyearthtools.pipeline.operations.dask.values.MaskValue)  |
|                      |                                      | - [ForceNormalised](pipeline_api.md#pyearthtools.pipeline.operations.dask.values.ForceNormalised)  |
| `pipeline.operations.dask.normalisation`  |                 | - [daskNormalisation](pipeline_api.md#pyearthtools.pipeline.operations.dask.normalisation.daskNormalisation)  |
|                      |                                      | - [Anomaly](pipeline_api.md#pyearthtools.pipeline.operations.dask.normalisation.Anomaly)  |
|                      |                                      | - [Deviation](pipeline_api.md#pyearthtools.pipeline.operations.dask.normalisation.Deviation)  |
|                      |                                      | - [Division](pipeline_api.md#pyearthtools.pipeline.operations.dask.normalisation.Division)  |
|                      |                                      | - [Evaluated](pipeline_api.md#pyearthtools.pipeline.operations.dask.normalisation.Evaluated)  |
| `pipeline.operations.dask.conversion`  |                    | - [ToXarray](pipeline_api.md#pyearthtools.pipeline.operations.dask.conversion.ToXarray)  |
|                      |                                      | - [ToNumpy](pipeline_api.md#pyearthtools.pipeline.operations.dask.conversion.ToXarray)  |
| `pipeline.operations.numpy`  |                              | - [Stack](pipeline_api.md#pyearthtools.pipeline.operations.numpy.Stack)  |
|                      |                                      | - [Concatenate](pipeline_api.md#pyearthtools.pipeline.operations.numpy.Concatenate)  |
| `pipeline.operations.numpy.augment`  |                      | - [Rotate](pipeline_api.md#pyearthtools.pipeline.operations.numpy.augment.Rotate)  |
|                      |                                      | - [Flip](pipeline_api.md#pyearthtools.pipeline.operations.numpy.augment.Flip)  |
|                      |                                      | - [Transform](pipeline_api.md#pyearthtools.pipeline.operations.numpy.augment.Transform)  |
| `pipeline.operations.numpy.filters`  |                      | - [NumpyFilter](pipeline_api.md#pyearthtools.pipeline.operations.numpy.filters.NumpyFilter)  |
|                      |                                      | - [DropAnyNan](pipeline_api.md#pyearthtools.pipeline.operations.numpy.filters.DropAnyNan)  |
|                      |                                      | - [DropAllNan](pipeline_api.md#pyearthtools.pipeline.operations.numpy.filters.DropAllNan)  |
|                      |                                      | - [DropValue](pipeline_api.md#pyearthtools.pipeline.operations.numpy.filters.DropValue)  |
|                      |                                      | - [Shape](pipeline_api.md#pyearthtools.pipeline.operations.numpy.filters.Shape)  |
| `pipeline.operations.numpy.reshape`  |                      | - [Rearrange](pipeline_api.md#pyearthtools.pipeline.operations.numpy.reshape.Rearrange)  |
|                      |                                      | - [Squeeze](pipeline_api.md#pyearthtools.pipeline.operations.numpy.reshape.Squeeze)  |
|                      |                                      | - [Expand](pipeline_api.md#pyearthtools.pipeline.operations.numpy.reshape.Expand)  |
|                      |                                      | - [Squeeze](pipeline_api.md#pyearthtools.pipeline.operations.numpy.reshape.Squeeze)  |
|                      |                                      | - [Flattener](pipeline_api.md#pyearthtools.pipeline.operations.numpy.reshape.Flattener)  |
|                      |                                      | - [Flatten](pipeline_api.md#pyearthtools.pipeline.operations.numpy.reshape.Flatten)  |
|                      |                                      | - [SwapAxis](pipeline_api.md#pyearthtools.pipeline.operations.numpy.reshape.SwapAxis)  |
| `pipeline.operations.numpy.select`  |                       | - [Select](pipeline_api.md#pyearthtools.pipeline.operations.numpy.select.Select)  |
|                      |                                      | - [Slice](pipeline_api.md#pyearthtools.pipeline.operations.numpy.select.Slice)  |
| `pipeline.operations.numpy.split`  |                        | - [OnAxis](pipeline_api.md#pyearthtools.pipeline.operations.numpy.split.OnAxis)  |
|                      |                                      | - [OnSlice](pipeline_api.md#pyearthtools.pipeline.operations.numpy.split.OnSlice)  |
|                      |                                      | - [VSplit](pipeline_api.md#pyearthtools.pipeline.operations.numpy.split.VSplit)  |
|                      |                                      | - [HSplit](pipeline_api.md#pyearthtools.pipeline.operations.numpy.split.HSplit)  |
| `pipeline.operations.numpy.values`  |                       | - [FillNan](pipeline_api.md#pyearthtools.pipeline.operations.numpy.values.FillNan)  |
|                      |                                      | - [MaskValue](pipeline_api.md#pyearthtools.pipeline.operations.numpy.values.MaskValue)  |
|                      |                                      | - [ForceNormalised](pipeline_api.md#pyearthtools.pipeline.operations.numpy.values.ForceNormalised)  |
| `pipeline.operations.numpy.normalisation`  |                | - [numpyNormalisation](pipeline_api.md#pyearthtools.pipeline.operations.numpy.normalisation.numpyNormalisation)  |
|                      |                                      | - [Anomaly](pipeline_api.md#pyearthtools.pipeline.operations.numpy.normalisation.Anomaly)  |
|                      |                                      | - [Deviation](pipeline_api.md#pyearthtools.pipeline.operations.numpy.normalisation.Deviation)  |
|                      |                                      | - [Division](pipeline_api.md#pyearthtools.pipeline.operations.numpy.normalisation.Division)  |
|                      |                                      | - [Evaluated](pipeline_api.md#pyearthtools.pipeline.operations.numpy.normalisation.Evaluated)  |
| `pipeline.operations.transform`  |                          | - [TimeOfYear](pipeline_api.md#pyearthtools.pipeline.operations.numpy.normalisation.TimeOfYear)  |
|                      |                                      | - [AddCoordinates](pipeline_api.md#pyearthtools.pipeline.operations.numpy.normalisation.AddCoordinates)  |
| `pipeline.samplers`  |                                      | - [EmptyObject](pipeline_api.md#pyearthtools.pipeline.samplers.EmptyObject)  |
| `pipeline.samplers`  |                                      | - [Sampler](pipeline_api.md#pyearthtools.pipeline.samplers.Sampler)  |
| `pipeline.samplers`  |                                      | - [Default](pipeline_api.md#pyearthtools.pipeline.samplers.Default)  |
| `pipeline.samplers`  |                                      | - [SuperSampler](pipeline_api.md#pyearthtools.pipeline.samplers.SuperSampler)  |
| `pipeline.samplers`  |                                      | - [Random](pipeline_api.md#pyearthtools.pipeline.samplers.Random)  |
| `pipeline.samplers`  |                                      | - [DropOut](pipeline_api.md#pyearthtools.pipeline.samplers.DropOut)  |
| `pipeline.samplers`  |                                      | - [RandomDropOut](pipeline_api.md#pyearthtools.pipeline.samplers.RandomDropOut)  |





