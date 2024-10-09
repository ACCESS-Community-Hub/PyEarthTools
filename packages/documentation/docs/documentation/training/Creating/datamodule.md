# Datamodule's

`edit.training.data` directly exposes a collection of data sources styled after `pytorch lightning's` datamodule which provide a seamless connection to `edit.pipeline` for data retrieval.

## Usage

```python
import edit.training
import edit.pipeline


edit.training.data.PipelineDataModule(
    edit.pipeline.Pipeline.sample()
)
```

The `PipelineDataModule` can also directly utilise train and validation splits to configure the pipelines to source the correct data for each part of training. This is given as a `edit.pipeline.Iterator`.

```python
datamodule = edit.training.data.PipelineDataModule(
    edit.pipeline.Pipeline.sample(),
    train_split = edit.pipeline.iterators.DateRange('2000', '2020', '6 hours'),
)
```

Using `.train()`, or `.eval()` switches the iterator between the two for usage in a training loop.

```python
datamodule.train()
for data in datamodule:
    model.forward(data)
```

## Default DataModule

While the `PipelineDataModule` provides basic access to pipeline for ML Usage, `default.PipelineDefaultDataModule` provides similar to pytorch lightning experience with the ability to control the batch size of returned data.

This can then be used in a training loop just like the base, but now with batches.

In the future, more options will be added, with `workers` being a priority to be added.

```python
datamodule = edit.training.data.default.PipelineDefaultDataModule(
    edit.pipeline.Pipeline.sample(),
    train_split = edit.pipeline.iterators.DateRange('2000', '2020', '6 hours'),
    batch_size = 16,
)
```

Ideally, this default module should work with any machine learning framework, but if not...


## Making your own DataModule

Any datamodule class should subclass from `PipelineDataModule`, it handles the parsing of the `pipelines` and the iteration split. 

The `pipelines` are then accessible from `.pipelines`, and can be a dictionary, a tuple, a pipeline or a dictionary of tuples. 
To assist in the use of this complex type variable, `PipelineDataModule.map_function_to_pipelines` is provided which automates the mapping of a function over the pipelines, and it returns in the same type and structure as the pipeline. 

Additionally, `map_function` is given which exposes the ability to map a function over any object just like a `pipeline`. `map_function_to_pipelines`, `train`, and `eval` all use the function to actually implement the functionality. 

This datamodule can be index, iterating over, or `get_sample` used to get a sample of data.

Check out the `default` and `lightning` modules for examples of an implementation.