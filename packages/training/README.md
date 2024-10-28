# EDIT Training

Using `edit.data` this package aims to significantly speed up ML model investigations and experimentations. 

With `edit.pipeline` a data stram can be configured to prepare the data for training or prediction. Once the model is wrapped in a `Wrapper`, it can be trained upon, or connected with a `Predictor` for inference.

## Available Frameworks

As wrappers are needed to connect the model to the rest of the system only the following frameworks are available, but it would be easy to add more,

| Name | Description |
| ---- | ----------- |
| Lightning | Pytorch Lightning - Managed pytorch code |
| ONNX | Complied models for prediction |
| XGBoost | Gradient boosted Trees |


## Usage

First create the datamodule

```python
import edit.training
import edit.pipeline

datamodule = edit.training.data.default.PipelineDefaultDataModule(
    edit.pipeline.Pipeline.sample(), 
    train_split = edit.pipeline.iterators.DateRange('2000-01-01T00', '2000-02-01T00', '1 day')
)
# Training Mode
datamodule.train()
```

Then create the model, and the model wrapper

```python

model = MODEL_GOES_HERE
model_wrapper = edit.training.wrapper.FRAMEWORK.Wrapper(model, datamodule)
```

If the framework supports it, run training

```python
model_wrapper.fit()
```

Or if only predictions are available, connect with a Prediction Wrapper
```python
predictor = edit.training.wrapper.predict.PREDICTION_TYPE(model_wrapper)
predictor.predict('2000-01-01T00')
```
