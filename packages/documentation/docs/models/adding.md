# Adding a Model

To add a new model accessible within `edit-models`, the user may subclass `edit.models.BaseForecastModel`.

Only `load` must be provided by the child class which is expected to return an `edit.training.EDIT_Inference` object, and a dictionary to pass to the `edit.training.MLIndex`.

If custom behaviour is needed for the `EDIT_Inference` (Wrapper) it may be subclassed and built out.

Additionally, if any other kwargs are needed for the child model, override the `__init__` function, and specify them there. Any of the `edit.models` commands will automatically detect those kwargs and prompt the user for them.

## Values

There are a set of properties which can be set in the registered model to alter the default behaviour.

### Setup

    - Setting `_default_config_path` provides a default config path. 
        This should be given, otherwise it must be set by the user each time.
    - Setting `_times` allows a model to specify which time deltas need to be 
        retrieved for predictions. Used for live download.
    - Setting `_download_paths` specifies files to download.
    - Setting `_name` provides the name of the model. It is best to set 
        this identical to where the model is registered to.
        If not given, will be the class name. Use '/' to set categories.

### `_download_paths`

    Setting `_download_paths` in the class will allow those assets to be automatically retrieved and stored.
    They are then accessible underneath a directory retrievable from `self.assets`.

    If given as a str the last '/' will be used as the name, or if given as a tuple, 
    the first element is the link, and the second the name.

    These paths can be to either a file or a zip file on a server or on the local machine.

    If the assets should be downloaded each time, set `_redownload_each_time` to True.

## Example

```python
@edit.models.register('MODEL/NAME')
class ModelRegister(edit.models.BaseForecastModel):
    """
    This doc will be printed on model load,
    so keep copyright, license or other info here

    \b
    Arguments:
        Helpful to list the argument here for the user
    """
    _default_config_path = SET_TO_WHERE_THE_DEFAULT_CONFIGS_ARE_KEPT
    _times = [0] # Data needed for running if download is needed
    _download_paths = [
        # Specify the files needed to run the model
    ]
    _name = 'MODEL/NAME'

    def __init__(self, pipeline: str, output: str | Path, *, extra_info_needed: int, **kwargs) -> None:
        ...

```

## Registeration

Once the model has been built, it can be registered, this can be done in two ways, entrypoints or decorators.

### Entrypoints

Using this method, allows the model to be auto discovered once it is installed with no other action needed.

This is done in the `pyproject.toml` file

```toml
[project.entry-points."edit.models.model"]
MODEL_NAME = "MODULE.registered_model:MODEL_CLASS"
```

### Decorators

This method requires the module to be imported before it is visible, but allows for easy extra registeration of in development models.

```python

@edit.models.register('MODEL/NAME')
class MODELRegister(edit.models.BaseForecastModel):
    ...
```

If using this method and wanting to use the commands, it is possible to set the module to be imported on commands running. This is done by setting `EDIT_MODELS_IMPORTS` in the environment.

Seperate modules by ':', and within each specification, split the name and path by '@'.

```shell
export EDIT_MODELS_IMPORTS=MODULE_NAME@PATH_TO_MODULE_GOES_HERE:
```
