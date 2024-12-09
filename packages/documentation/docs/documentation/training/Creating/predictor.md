# Predictor

By themselves `ModelWrapper` provide only the most basic functionality, loading models, saving and running a single forward pass with the model.

Using `pyearthtools.training.Predictor` allows for a managed prediction to be run with the model, taking an index in which to get data from the `Pipeline` setup with the model. 

This however, simply uses the `ModelWrapper.predict` function so runs a single forward pass.

## Reverse

As tensors or numpy arrays are not that useful in of themselves, the output of the model is automatically run through a reverse pipeline. This can alter the shape and ideally convert it back into a more useful dataset / netcdf object. 

The pipeline that is used can be configured with `reverse_pipeline`. It can either be a particular pipeline within the data or a whole new pipeline. 

## Hooks
Additionally, a hook of `after_predict` can be implemented to alter the data after the prediction and reversal is run.


## Other Predictors

As the base `Predictor` simply expects the model to do the work, other predictors can be implemented to handle more complex logic, i.e.
- Temporal prediction
- Coupled modelling


### Timeseries Predictor

As a common pattern in the field is to run the model recurrently feeding the outputs back in as the inputs, predictors are provided to handle this.

`TimeSeriesPredictor` is base class for this exposing the function `recurrent`.

| Predictor | Description |
| --------- | ----------- |
| `TimeSeriesPredictor` | Base temporal series predictor |
| `ManualTimeSeriesPredictor` | Expects the model to handle the recurrent prediction internally | 
| `TimeSeriesAutoRecurrentPredictor` | Autorecurrent predictor, `inputs==outputs`.|
| `TimeSeriesManagedPredictor` | Managed predictor to get missing data from `datamodule`. Best used if `datamodule` is a dictionary, and for models with `prognostics`, `forcings` / `diagnostics`.|


See the docs for each the predictors for more information on how to use it, and what hooks it provides.
