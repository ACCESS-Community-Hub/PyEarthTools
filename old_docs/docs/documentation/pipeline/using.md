# Using a pipeline

A `Pipeline` can be used in three primary ways,

| Type | Code|
|----|----|
| Direct Indexing | `pipeline[idx]` |
| Iteration | `for i in pipeline` |
| Applying | `pipeline.apply` |

## Direct Indexing
A pipeline just needs to consist of a sequence of steps following an `index` for this method to work.
The index is passed directly to the highest `Index` object, which can either be a `pyearthtools.data.Index` or `pyearthtools.pipeline.PipelineIndex`. 

Therefore, if a cache is present, it will be used.

```python
import pyearthtools.pipeline

pipeline = pyearthtools.pipeline.Pipeline.sample()
pipelines['2000-01-01T00']
```

## Applying
A pipeline can also be used to apply its modifications to any data sample, with `.apply()`.

```python
import pyearthtools.pipeline

data_sample = DATA

pipeline = pyearthtools.pipeline.Pipeline(
    pyearthtools.pipeline.operations.*
)
pipelinesa.apply(data_sample)
```

## Iteration
In order to iterate over the pipeline, the iterator needs to be set. This can be done in the `__init__` function or by setting `pipeline.iterator`.

These must be a `pyearthtools.pipeline.Iterator` of which basics are provided:

| Name | Description |
| ---- | ----------- |
| Range | Iterate through a `range` |
| PreDefined | Yield elements from a predefined iterable |
| File | Iterate over elements in a file, each line treated separately | 
| DateRange | Using `pyearthtools.data.TimeRange` iterate over dates |
| DateRangeLimit | Starting at date and with interval, yield a number of samples |
| Randomise | Randomise another `Iterator` |
| SuperIterator | Iterate over a sequence of `Iterator`'s |

If a tuple is provided, it will be auto wrapped in a `SuperIterator`

### Example

```python
import pyearthtools.pipeline
import pyearthtools.data

pipeline = pyearthtools.pipeline.Pipeline(
    pyearthtools.data.archive.ERA5.sample(),
    iterator = pyearthtools.pipeline.iterators.DateRange('2000-01-01T00', '2010-01-01T00', '6 hours'),
) 

or

pipeline.iterator = pyearthtools.pipeline.iterators.DateRange('2000-01-01T00', '6 hours', 1000)
```

## Filters

Adding filters to check the validity of samples into the pipeline will run the check on the iteration, and if invalid will be automatically skipped, raising a warning every `pipeline_V2.exceptions.max_filter` which defaults to `10`.

This is useful to filter out `nan` values or invalid `shapes`.

These filters are available under `pyearthtools.pipeline.operations.[DATA_TYPE].filters`, where `DATA_TYPE` can be `xarray`, `numpy`, or `dask` currently.

## Sampling

In addition to setting the `iterator` of a pipeline, it can be useful to set the sampling strategy. This provides a way to modify the flow of samples out of the pipeline after they have been retrieved with the `index` from the `iterator`. This can be done in the `__init__` function or by setting `pipeline.sampler`.

| Name | Description |
| ---- | ----------- |
| Default | Default sampler, does nothing | 
| SuperSampler | Combines multiple samplers |
| Random | Build a buffer and randomly sample from it |
| DropOut | Drop out samples at a set interval |
| RandomDropOut | Randomly drop out samples |

If a tuple is provided, it will be auto wrapped in a `SuperSampler`


### Example

```python
import pyearthtools.pipeline
import pyearthtools.data

pipeline = pyearthtools.pipeline.Pipeline(
    pyearthtools.data.archive.ERA5.sample(),
    sampler = pyearthtools.pipeline.samplers.Random(buffer_len = 100),
) 

or

pipeline.sampler = pyearthtools.pipeline.samplers.RandomDropOut(chance = 50.0)
```