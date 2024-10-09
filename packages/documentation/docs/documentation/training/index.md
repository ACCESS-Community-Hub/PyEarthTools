# Training

`edit.training` utilises the other components of `edit` to provide the data and structures in which to train machine learning models.

## Data

Particularly, `edit.training` provides `PipelineDataModule`'s to use pipelines as a source of data.

```python
import edit.training
import edit.pipeline


edit.training.data.PipelineDataModule(
    edit.pipeline.Pipeline.sample()
)
```

The `PipelineDataModule` can also directly utilise train and validation splits to configure the pipelines to source the correct data for each part of training. This is given as a `edit.pipeline.Iterator`.

```python
edit.training.data.PipelineDataModule(
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

Additionally, data modules can be configured for the requirements of particular ML frameworks, 

Currently, the following are implemented,

| Framework | Path | Info |
| --------- | ---- | ---- |
| Lightning | `edit.training.data.lightning` | Pytorch Lightning data module |
| Other | `edit.training.data.default` | Default data module to mimic lightning for any other framework |

`pipeline`'s can be given as either `Pipeline`, tuple of `Pipeline`'s or dictionary of `Pipeline`'s.

## Wrapper

Loading data is only part of the problem, it actually needs to be connected into the model, and training ran.

In `edit.training` this takes the form of wrappers, particularly, `ModelWrapper` and `TrainWrapper`.

A `ModelWrapper` is the base wrapper class which provides the connection between a model and it's data source. 
By itself it cannot do much but provides the template in which other frameworks can be connected.

A `TrainWrapper` simply provides the interface to run `fit`, and is the responsibility of the underlying implementation to provide.

```python
import edit.training
import edit.pipeline

data = edit.training.data.PipelineDataModule(
    edit.pipeline.Pipeline.sample()
)

model = edit.training.ModelWrapper(ML_MODEL, data)
```

### Frameworks 

The following frameworks have been implemented

| Framework | Training | Inference |
| --------- | :------: | :-------: |
| Lightning - Pytorch | &#9745; | &#9745; |
| XGBoost | &#9745; | &#9745; |
| Onnx | &#9744; | &#9745; |

### Prediction 

A `Predictor` provides the implementation in which to run prediction, it utilises the underlying data source to retrieve the initial conditions, and then the models implemented `predict` function to run the prediction.

Additionally, this `Predictor` can be subclassed to provide further logic, for example, a time series aware prediction.

For models which predict into the future, a `TimeSeriesRecurrentPredictor` can be used which allows for rollout of predictions.

