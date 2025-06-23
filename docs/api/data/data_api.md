# Data API Docs

## `data.archive`

```{eval-rst}
.. autoclass:: pyearthtools.data.archive.ZarrIndex
    :members:

.. autoclass:: pyearthtools.data.archive.ZarrTimeIndex
    :members:

.. autofunction:: pyearthtools.data.archive.extensions.register_archive
.. autofunction:: pyearthtools.data.archive.reset_root
.. autofunction:: pyearthtools.data.archive.set_root
.. autofunction:: pyearthtools.data.archive.config_root

```

## `data.derived`

```{eval-rst}
.. autoclass:: pyearthtools.data.derived.DerivedValue
    :members:

.. autoclass:: pyearthtools.data.derived.TimeDerivedValue
    :members:

.. autoclass:: pyearthtools.data.derived.AdvancedTimeDerivedValue
    :members:

.. autoclass:: pyearthtools.data.derived.Insolation
    :members:

```

## `data.download`

```{eval-rst}
.. autoclass:: pyearthtools.data.download.arcoera5.ARCOERA5
    :members:
.. autodata:: pyearthtools.data.download.arcoera5.LEVELS
.. autodata:: pyearthtools.data.download.arcoera5.LONG_NAMES
    :no-value:
.. autodata:: pyearthtools.data.download.arcoera5.SHORT_NAMES
    :no-value:
```

## `data.indexes`

```{eval-rst}
.. autoclass:: pyearthtools.data.indexes.Index
    :members:

.. autoclass:: pyearthtools.data.indexes.DataIndex
    :members:

.. autoclass:: pyearthtools.data.indexes.FileSystemIndex
    :members:

.. autoclass:: pyearthtools.data.indexes.TimeIndex
    :members:

.. autoclass:: pyearthtools.data.indexes.TimeDataIndex
    :members:

.. autoclass:: pyearthtools.data.indexes.AdvancedTimeIndex
    :members:

.. autoclass:: pyearthtools.data.indexes.AdvancedTimeDataIndex
    :members:

.. autoclass:: pyearthtools.data.indexes.BaseTimeIndex
    :members:

.. autoclass:: pyearthtools.data.indexes.DataFileSystemIndex
    :members:

.. autoclass:: pyearthtools.data.indexes.ArchiveIndex
    :members:

.. autoclass:: pyearthtools.data.indexes.ForecastIndex
    :members:

.. autoclass:: pyearthtools.data.indexes.StaticDataIndex
    :members:

.. autoclass:: pyearthtools.data.indexes.CachingIndex
    :members:

.. autoclass:: pyearthtools.data.indexes.CachingForecastIndex
    :members:

.. autoclass:: pyearthtools.data.indexes.IntakeIndex
    :members:

.. autoclass:: pyearthtools.data.indexes.IntakeIndexCache
    :members:

.. autoclass:: pyearthtools.data.indexes.cacheIndex.BaseCacheIndex
    :members:

.. autoclass:: pyearthtools.data.indexes.cacheIndex.MemCache
    :members:    
.. autoclass:: pyearthtools.data.indexes.cacheIndex.FileSystemCacheIndex
    :members:    
.. autoclass:: pyearthtools.data.indexes.cacheIndex.CacheFactory
    :members:        
.. autoclass:: pyearthtools.data.indexes.cacheIndex.FunctionalCache
    :members:            

.. autoclass:: pyearthtools.data.indexes.combine.InterpolationIndex
    :members:                

.. autoclass:: pyearthtools.data.indexes.fake.FakeIndex
    :members:                

.. autofunction:: pyearthtools.data.indexes.extensions.register_accessor    
```

## `data.modifications`

```{eval-rst}
.. autoclass:: pyearthtools.data.modifications.Modification
    :members:                

.. autofunction:: pyearthtools.data.modifications.register_modification
.. autofunction:: pyearthtools.data.modifications.variable_modifications

.. autoclass:: pyearthtools.data.modifications.aggregations.Aggregation
    :members:                
.. autoclass:: pyearthtools.data.modifications.aggregations.AggregationGeneral
    :members:                
.. autoclass:: pyearthtools.data.modifications.aggregations.Mean
    :members:                    
.. autoclass:: pyearthtools.data.modifications.aggregations.Accumulate
    :members:                        

.. autoclass:: pyearthtools.data.modifications.constants.Constant
    :members:                            

.. autoclass:: pyearthtools.data.modifications.decorator.VariableModification
    :members:

.. autoclass:: pyearthtools.data.modifications.decorator.Modifier
    :members:    

.. autoclass:: pyearthtools.data.modifications.reductions.Reduction
    :members:    
.. autoclass:: pyearthtools.data.modifications.reductions.Groupby
    :members:
.. autofunction:: pyearthtools.data.modifications.reductions.Hourly
.. autofunction:: pyearthtools.data.modifications.reductions.Daily
.. autofunction:: pyearthtools.data.modifications.reductions.Monthly    
.. autofunction:: pyearthtools.data.modifications.register.register_modification
```

## `data.operations`

```{eval-rst}
.. autofunction:: pyearthtools.data.operations.percentile
.. autofunction:: pyearthtools.data.operations.aggregation
.. autofunction:: pyearthtools.data.operations.binning

.. autoclass:: pyearthtools.data.operations.SpatialInterpolation
    :members:

.. autoclass:: pyearthtools.data.operations.TemporalInterpolation
    :members:

.. autoclass:: pyearthtools.data.operations.FullInterpolation
    :members:

.. autofunction:: pyearthtools.data.operations.index_routines.series
.. autofunction:: pyearthtools.data.operations.index_routines.safe_series
.. autofunction:: pyearthtools.data.operations.index_operations.split_ds
.. autofunction:: pyearthtools.data.operations.index_operations.split_ds_gen
.. autofunction:: pyearthtools.data.operations.index_operations.aggregation
.. autofunction:: pyearthtools.data.operations.index_operations.find_range
.. autofunction:: pyearthtools.data.operations.utils.identify_time_dimension

.. autofunction:: pyearthtools.data.operations.forecast_op.forecast_series
.. autofunction:: pyearthtools.data.operations.forecast_op.forecast_as_basetime
.. autofunction:: pyearthtools.data.operations.forecast_op.forecast_select_time
```

## `data.patterns`

```{eval-rst}
.. autoclass:: pyearthtools.data.patterns.PatternIndex
    :members:

.. autoclass:: pyearthtools.data.patterns.PatternTimeIndex
    :members:    

.. autoclass:: pyearthtools.data.patterns.PatternForecastIndex
    :members:    

.. autoclass:: pyearthtools.data.patterns.PatternVariableAware
    :members:    

.. autoclass:: pyearthtools.data.patterns.Argument
    :members:        

.. autoclass:: pyearthtools.data.patterns.ArgumentExpansion
    :members:    

.. autoclass:: pyearthtools.data.patterns.ArgumentExpansionVariable
    :members:    

.. autoclass:: pyearthtools.data.patterns.ArgumentExpansionFactory
    :members:    

.. autoclass:: pyearthtools.data.patterns.Direct
    :members:    

.. autoclass:: pyearthtools.data.patterns.TemporalDirect
    :members:    

.. autoclass:: pyearthtools.data.patterns.ForecastDirect
    :members:

.. autoclass:: pyearthtools.data.patterns.DirectVariable
    :members:

.. autoclass:: pyearthtools.data.patterns.ForecastDirectVariable
    :members:

.. autoclass:: pyearthtools.data.patterns.TemporalDirectVariable
    :members:

.. autoclass:: pyearthtools.data.patterns.DirectFactory
    :members:

.. autoclass:: pyearthtools.data.patterns.ExpandedDate
    :members:

.. autoclass:: pyearthtools.data.patterns.TemporalExpandedDate
    :members:

.. autoclass:: pyearthtools.data.patterns.ForecastExpandedDate
    :members:

.. autoclass:: pyearthtools.data.patterns.ExpandedDateVariable
    :members:    

.. autoclass:: pyearthtools.data.patterns.ForecastExpandedDateVariable
    :members:    

.. autoclass:: pyearthtools.data.patterns.TemporalExpandedDateVariable
    :members:    

.. autoclass:: pyearthtools.data.patterns.ExpandedDateFactory
    :members:    

.. autoclass:: pyearthtools.data.patterns.Static    
    :members:

.. autoclass:: pyearthtools.data.patterns.ParsingPattern    
    :members:    

.. autoclass:: pyearthtools.data.patterns.ZarrIndex    
    :members:    

.. autoclass:: pyearthtools.data.patterns.ZarrTimeIndex    
    :members:    
```

## `data.save`

```{eval-rst}
.. autofunction:: pyearthtools.data.save.save
.. autoclass:: pyearthtools.data.save.ManageFiles    
    :members:   
.. autoclass:: pyearthtools.data.save.ManageTemp    
    :members:       

.. autofunction:: pyearthtools.data.save.array.save
.. autofunction:: pyearthtools.data.save.dask.save    
.. autofunction:: pyearthtools.data.save.dataset.save
.. autofunction:: pyearthtools.data.save.dataset.to_netcdf
.. autofunction:: pyearthtools.data.save.dataset.to_zarr
.. autofunction:: pyearthtools.data.save.jsonsave.save
.. autofunction:: pyearthtools.data.save.plot.save

.. autofunction:: pyearthtools.data.save.save_utils.check_if_exists
.. autofunction:: pyearthtools.data.save.save_utils.make_new_filename

.. autoclass:: pyearthtools.data.save.save_utils.keep_clear    
    :members:       
```

## `data.transforms`

```{eval-rst}
.. autoclass:: pyearthtools.data.transforms.Transform
    :members:  
.. autoclass:: pyearthtools.data.transforms.TransformCollection
    :members:      
.. autoclass:: pyearthtools.data.transforms.FunctionTransform
    :members:    
.. autoclass:: pyearthtools.data.transforms.Derive
    :members:        

.. autofunction:: pyearthtools.data.transforms.aggregation.over
.. autofunction:: pyearthtools.data.transforms.aggregation.leaving    

.. autoclass:: pyearthtools.data.transforms.aggregation.Aggregate
    :members:  

.. autoclass:: pyearthtools.data.transforms.attributes.SetAttributes
    :members:  
.. autoclass:: pyearthtools.data.transforms.attributes.SetEncoding
    :members:      
.. autoclass:: pyearthtools.data.transforms.attributes.SetType
    :members:          
.. autoclass:: pyearthtools.data.transforms.attributes.Rename
    :members:          

.. autofunction:: pyearthtools.data.transforms.coordinates.get_longitude
.. autofunction:: pyearthtools.data.transforms.coordinates.weak_cast_to_int

.. autoclass:: pyearthtools.data.transforms.coordinates.StandardLongitude
    :members:  
.. autoclass:: pyearthtools.data.transforms.coordinates.ReIndex
    :members:      
.. autoclass:: pyearthtools.data.transforms.coordinates.StandardCoordinateNames
    :members:          
.. autoclass:: pyearthtools.data.transforms.coordinates.Select
    :members:              
.. autoclass:: pyearthtools.data.transforms.coordinates.Drop
    :members:    
.. autoclass:: pyearthtools.data.transforms.coordinates.Flatten
    :members:
.. autoclass:: pyearthtools.data.transforms.coordinates.Expand
    :members:
.. autoclass:: pyearthtools.data.transforms.coordinates.SelectFlatten
    :members:
.. autoclass:: pyearthtools.data.transforms.coordinates.Assign
    :members:    
.. autoclass:: pyearthtools.data.transforms.coordinates.Pad
    :members:        

.. autofunction:: pyearthtools.data.transforms.default.get_default_transforms

.. autofunction:: pyearthtools.data.transforms.derive.evaluate
.. autofunction:: pyearthtools.data.transforms.derive.derive_equations

.. autoclass:: pyearthtools.data.transforms.dimensions.StandardDimensionNames
    :members:  

.. autoclass:: pyearthtools.data.transforms.dimensions.Expand
    :members:      

.. autoclass:: pyearthtools.data.transforms.interpolation.Interpolate
    :members:     
.. autoclass:: pyearthtools.data.transforms.interpolation.XESMF
    :members:         
.. autoclass:: pyearthtools.data.transforms.interpolation.InterpolateNan
    :members:         

.. autofunction:: pyearthtools.data.transforms.interpolation.like

.. autoclass:: pyearthtools.data.transforms.mask.UnderlyingMaskTransform
    :members:     
.. autoclass:: pyearthtools.data.transforms.mask.Dataset
    :members:
.. autoclass:: pyearthtools.data.transforms.mask.Replace
    :members:             

.. autoclass:: pyearthtools.data.transforms.optimisation.Rechunk
    :members:     

.. autofunction:: pyearthtools.data.transforms.region.check_shape
.. autofunction:: pyearthtools.data.transforms.region.order
.. autofunction:: pyearthtools.data.transforms.region.like

.. autoclass:: pyearthtools.data.transforms.region.Bounding
    :members: 
.. autoclass:: pyearthtools.data.transforms.region.Select
    :members:     
.. autoclass:: pyearthtools.data.transforms.region.ISelect
    :members:     

.. autofunction:: pyearthtools.data.transforms.region.PointBox
.. autofunction:: pyearthtools.data.transforms.region.Lookup   
.. autofunction:: pyearthtools.data.transforms.region.Geosearch   

.. autoclass:: pyearthtools.data.transforms.region.ShapeFile
    :members:     

.. autofunction:: pyearthtools.data.transforms.utils.parse_dataset

```

## `data.catalog`

```{eval-rst}
.. autoclass:: pyearthtools.data.catalog.Catalog
    :members:
.. autoclass:: pyearthtools.data.catalog.CatalogEntry
    :members:

.. autofunction:: pyearthtools.data.catalog.get_name    
```

## `data.collection`

```{eval-rst}
.. autoclass:: pyearthtools.data.collection.Collection
    :members:
.. autoclass:: pyearthtools.data.collection.LabelledCollection
    :members:    
```

## `data.exceptions`
```{eval-rst}
.. autoclass:: pyearthtools.data.exceptions.InvalidIndexError
    :members:
.. autoclass:: pyearthtools.data.exceptions.InvalidDataError
    :members:
.. autoclass:: pyearthtools.data.exceptions.DataNotFoundError
    :members:    
```

## `data.load`
```{eval-rst}
.. autofunction:: pyearthtools.data.load.load
```

## `data.time`
```{eval-rst}
.. autofunction:: pyearthtools.data.time.multisplit
.. autofunction:: pyearthtools.data.time.find_components
.. autofunction:: pyearthtools.data.time.strip_to_common_resolution
.. autofunction:: pyearthtools.data.time.time_delta
.. autofunction:: pyearthtools.data.time.time_delta_resolution
.. autofunction:: pyearthtools.data.time.range_samples

.. autoclass:: pyearthtools.data.time.range_samples
    :members:    
.. autoclass:: pyearthtools.data.time.Petdt
    :members:        
.. autoclass:: pyearthtools.data.time.TimeDelta
    :members:        
.. autoclass:: pyearthtools.data.time.TimeRange
    :members:            
```

## `data.warnings`
```{eval-rst}
.. autoclass:: pyearthtools.data.warnings.pyearthtoolsDataWarning
    :members:
.. autoclass:: pyearthtools.data.warnings.IndexWarning
    :members:
.. autoclass:: pyearthtools.data.warnings.AccessorRegistrationWarning
    :members:    
```

