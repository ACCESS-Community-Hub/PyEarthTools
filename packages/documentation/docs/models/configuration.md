# Config Conventions

## Folder

The config folder for a model can use the following conventions to ease in setup

```txt
`Data/`
    - Location for all data loaders
`Pipeline/`
    - Location for all pipelines
```

It is assumed that most data configs will have a pipeline identically named to them for loading and preparing the data, however, the following exception applies.
If a `Data` config has a `()`, with a str inside, it represents a different data source, but the same pipeline, this is primarily used for `Live` and `Archived` data.

Additionaly, any data with a `-` represents an ancillary source, i.e. forcings and will not be included in the available data sources. Any text prior to the `-` represents the parent source and any after is it's purpose.
Getting `ancillary_pipeline` will give back a dictionary of ancillary pipelines associated with the chosen source.

### Examples

Consider the following structure.

```txt
├── Data
│   ├── ERA5-Forcings.yaml
│   ├── ERA5(Live)-Forcings.yaml
│   ├── ERA5(Live).yaml
│   └── ERA5.yaml
└── Pipeline
    └── ERA5.yaml
```

A user can request either `ERA5` or `ERA5(Live)` as the data source, these two sources are then loaded and use `Pipeline/ERA5.yaml` as it's pipeline.

When getting `ancillary_pipeline`, either `ERA5-Forcings` or `ERA5(Live)-Forcings` will be used, dependent on the data source as detailed above. If a `Pipeline/ERA5-Forcings.yaml` existed, both sources would then use this as their pipeline.

## Making Your Own

Using the above outlined structure, custom user configs can be made and stored outside of the model itself.
Passing `--config_path PATH` to any of the commands in `edit-models` will allow those config paths to be discovered and selected.

### Environmental Variable

An environment can define a list of paths at `EDIT_MODELS_CONFIGS`. These will be added to the valid pipelines, with the model class name added to the end.
For most models this should be the full categorical path of the model, see each model for it's `_name`. If not set will be the class name.

### Custom Config Example

To demonstrate how this works, we show an example of how to adjust the Spherical Fourier Neural Operator (SFNO) data config to initialise the model with a 2K increase in temperature.

As each model has a predefined configuration, it may be easier to simply piggyback off of the existing setup, and add an `edit` transform to modify the data. For example, a `derive` transform could be added to modify the data.

#### Data Config File

Upon loading `EDIT_MODELS_DEFAULT_CONFIG` is automatically set to the the `_default_model_config` if given, therefore, it can be used to reference the original configs.

Using the provided `ERA5` data config included with the model, we can add a transform operation.

A `CachingIndex` is likely needed after due to a limitation in the current implementation of `edit.pipeline`.

```yaml
file: __EDIT_MODELS_DEFAULT_CONFIG__/Data/ERA5.yaml

# Now using derive, we can add a change.
edit.pipeline.operations.transforms.operation.TransformOperation:
  transforms:
    edit.data.transforms.derive.derive:
      t:
      - t + 2
      - EXPERIMENT_DETAIL: 'Added 2 degrees'
      2t:
      - 2t + 2
      - EXPERIMENT_DETAIL: 'Added 2 degrees'
edit.pipeline.indexes.cache.CachingIndex:
  cache: 'temp'
```

This change will add 2 degrees to both temperature fields, allowing a test under these initial conditions

This config must be saved into a directory `Data`, so that it can be identified by `edit-models` as a data config.

If using `$EDIT_MODELS_CONFIGS` to include this config, the folder path must look like the following, with the full model name a part of the path:

```txt
$MODEL_CATEGORICAL_PATH/Data/$CONFIG_NAME.yaml
$MODEL_CATEGORICAL_PATH/Pipeline/$CONFIG_NAME.yaml # If including a pipeline
```

Otherwise, the parent directories can be decided by the user.

#### Running a prediction with the config

Now with this config saved at `Experiments/Global/sfno/Data/ERA5(Climate).yaml`, the following command can be run. Note how the only change is the `--config_path` from a normal prediction

```shell
edit-models predict sfno --pipeline 'ERA5(Climate)' --output OUTPATH --time TIME --lead_time LEADTIME --config_path ./Experiments/Global/sfno
```

or

```shell
edit-models interactive --config_path ./Experiments/Global/sfno
```

or

```shell
export EDIT_MODELS_CONFIGS=./Experiments

edit-models predict sfno --pipeline 'ERA5(Climate)' --output OUTPATH --time TIME --lead_time LEADTIME 
```

or

```shell
export EDIT_MODELS_CONFIGS=./Experiments

edit-models interactive 
```

### Further use

With this basic template in mind, a user can adjust any data or pipeline file to run experiments, or add a new data and pipeline file to allow a new data source.

## Variable Assignment

Some data sources or pipelines may require variables to be set as they are loaded, i.e. to specifiy the ensemble member of a probablistic data source.

To set these variables, a trailing `{}` can be supplied to any pipeline selection.

### Variable Example

Consider a pipeline needing a MEMBER variable to be supplied, selecting the `ACCESS` pipeline the variable can be set via,

```txt
--pipeline ACCESS{MEMBER=member_id_goes_here}
```

### Including a variable in a pipeline

If you are writing a pipeline and want an assignment variable there, surround the key by `__`, so `__KEY__`,

i.e.

```yaml
edit.pipeline.operations.transforms.operation.TransformOperation:
  transforms:
    edit.data.transforms.derive.derive:
      t:
      - t + 2
      - EXPERIMENT_DETAIL: __NAME__
```

This will now allow a user to set `{NAME=name_goes_here}` to change this variable inside the pipeline.

### Default Values

If ':' follows the KEY part and still within '__*__', anything following will be considered the default value.

#### Example

This allows the modification factor to be set dynamically, and if not given, default to `2`.

```yaml
edit.pipeline.operations.transforms.operation.TransformOperation:
  transforms:
    edit.data.transforms.derive.derive:
      t:
      - t + __FACTOR:2__
      - EXPERIMENT_DETAIL: __NAME__
```
