# Python Package

`pyearthtools.models` exists as a python package, and can be used to run predictions either from the command line or within a python instance.

## Usage

`pyearthtools.models` provides the following top level functions for easy usage in a python environment.

### `.data`

This function allows the user to get te data that will be provided to the model for sanity checking or prepreparation.

Particularly, as some pipelines require the downloading of data from another source, it may prove helpful to call this function first, in a internet accessible location, cache it to a cache directory, and then run the prediction at a later point.

???+ note "Example"
    ```python
    import pyearthtools.models

    data = pyearthtools.models.data('model_name_here', 'pipeline_name_here', *args, **kwargs)
    ```

### `.predict`

Run predictions.

All arguments, such as the model, pipeline and directories must be fully specified.

This function will then import the model, download the assets if needed, and then run the prediction.

???+ note "Example"
    ```python
    import pyearthtools.models

    predictions = pyearthtools.models.predict('model_name_here', 'pipeline_name_here', *args, **kwargs)
    ```

### `.interactive`

Run predictions interactively.

This function is effectively a wrapper around `.predict` but will prompt the user for any missing arguments.

???+ note "Example"
    ```python
    import pyearthtools.models

    predictions = pyearthtools.models.interactive(model = 'model_name_here')
    predictions = pyearthtools.models.interactive()
    Which model would you like to use? [MODELS], (Key: model): USER_INPUT_HERE
    ```

## Direct Model Usage

Any installed and imported model will be accessible underneath `pyearthtools.models.*` for direct usage.

```python
import sfno
import pyearthtools.models

pyearthtools.models.SFNO() # Exists

```
