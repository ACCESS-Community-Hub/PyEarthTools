# XGBoost User Story

Mathilde Ritman – details of adding XGBoost to `pyearthtools`

July 2023

## Adding New Models

To add an XBoost model capability to `pyearthtools`, both the model architecture file and a customised model trainer were required. While unsupported models will always require a model architecture file to be written so that they can be implemented, customised model trainers are only required if existing trainers cannot support the new model class. This was the case for XGBoost, as at the time of implementation model trainers only supported PyTorch models.

## Setting up an XGBoost Model

The first step of implementing the XGBoost architecture in `pyearthtools` was to create a new class, called `XGBoost`. This class is a wrapper for the python package implementation of XGBoost, `xgboost`. The xgboost python package can implement an XGBoost regression algorithm as well as an XGBoost classifier. As such, the initialisation function was designed so that the user can specify the preferred XGBoost implementation, as well as pass model parameters to the chosen model.

Similarly, the XGBoost class offers the user the ability to implement the model using dask, allowing them to specify relevant configurations (including the number of workers and number of chunks for the input data) (see example below).

``` python title="XGBoost Model"

class XGBoost():

    def __init__(self,
        model_params: dict = {},
        use_dask: bool = False,
        n_workers: int=12,
    ):
        
        if use_dask:
            client = Client(n_workers=n_workers,)
            model = xgboost.dask.DaskXGBRegressor(**model_params)
```

After constructing the initialisation function, the trainer for the algorithm itself was added to the class. In the case of `XGBoost`, this involved calling the xgboost `.fit()` function and passing it the relevant training data and user specified model parameter choices (e.g., the loss function). Additional functionality was added to transform the training data appropriately for use with dask or the classifier model if these implementations had been specified.

### Adding custom loss functions

Customised loss (or 'objective') functions are often desired by users of machine learning. Implementing these for a model in `pyearthtools` can be achieved with ease. For XGBoost, a customised loss function was written, defined as the equally weighted linear combination of the L1 and L2 loss functions. The underlying python implementation, xgboost, expects objective functions of the form:

``` python title="Objective"

def objective(y_pred, y_true):

	…
    return gradient, hessian

```

Following this requirement, the customised loss function was written and saved to the `pyearthtools_xgboost` module. To allow the user to specify the customised loss function using the `pyearthtools` configuration file, functionality was added to the XGBoost class initialisation function (see below). In this case, the user specifies the path to the chosen loss function and the initialisation function uses the given string to find and pass the associated function to xgboost.

``` python title="Custom Loss"

if 'custom_loss' in model_params['objective'] and isinstance(model_params['objective'], str):

    obj = model_params['objective'].replace('custom_loss.','')
    custom_func = getattr(custom_loss, obj) # Get the callable function referenced by the provided string

    if not callable(custom_func):
        raise TypeError()

    model_params['objective'] = custom_func

```

## Writing a custom trainer

First, a new class was defined called `pyearthtoolsXGBoostTrainer` that inherits from the trainer template class [`pyearthtoolsTrainer`][pyearthtools.training.pyearthtoolsTrainer]. The trainer class comprises of four key functions required for model training within `pyearthtools`, namely, functions to perform model fitting, prediction, loading and saving (see template below). Each of these functions are essentially wrappers for the underlying xgboost implementation, some providing additional functionality for user specified training approaches.

Specifically, the [`pyearthtoolsTrainer`][pyearthtools.training.pyearthtoolsTrainer] provides the [`predict`][pyearthtools.training.pyearthtoolsTrainer.predict] function to be called by the user, obfuscating data loading and reconstruction logic, and relys on the underlying class to implement the `_predict_from_data` function which will run the prediction on passed data.

``` python title="Template"

import xgboost

from pyearthtools.training.trainer.template import pyearthtoolsTrainer
from pyearthtools.training.data.templates import DataStep

class pyearthtoolsXGBoostTrainer(pyearthtoolsTrainer):
    def __init__(self, 
                 model, 
                 train_data: DataStep, 
                 valid_data: DataStep = None, 
                 path: str | Path = None, 
                 **kwargs
                 ) -> None:
        super().__init__(model, train_data, valid_data, path)

        # Initialise Model
        self.model = model

        self.path = Path(path)

    def fit(self,):
    
    def _predict_from_data(self,):
        # Handle Predictions

    def load(self,):
        # Load saved model

    def save(self,):
        # Save model
    
    def eval(self,):
        # Evaluate model
```

The model fitting function operates in user-specified batches. The fitter iterates through the number of batches specified, and implements the XGBoost model class fitting function once per iteration. The fitter also allows the user to specify whether an existing saved model should be loaded and trained further.

The predictor offers functionality within the `pyearthtools` pipeline by wrapping the xgboost predict function to ensure that the returned data is the same shape of the input data where both are tuples of type (features, target).

Both the model load and save functions are simple xgboost wrappers, these will find or save a .json model at the specified path (see example below).

```python title="Example pyearthtools xgboost wrapper"

    def save(self, path: str | Path = None):
        # Save model
        
        if path is None:
            path = Path(self.path)

        self.model.save_model(path / "model.json")
```

In addition to the four key functions, a model evaluation function was written to provide quick summary statistics of model performance. The statistics are calculated over one batch of the user-specified training and validation datasets. The number of samples drawn from the batch can be limited by the user. On this data, globally-aggregated summary statistics are derived for regression prediction tasks, including mean absolute error, root mean squared error, bias and correlation. The statistics are saved to a `.json` file at the spacified experiment path. On the sampled validation dataset, a plot of the distribution of model errors is also produced and saved.
