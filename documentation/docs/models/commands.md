# Commands

```shell
[~]$ edit-models 
Usage: edit-models [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  predict      Run Prediction of registered model
  interactive  Interactive Prediction
  models       Print available models.
  data         Download data.
```

## Predict

```shell
[~]$ edit-models predict --help
Usage: edit-models predict [OPTIONS] MODEL

  Run Prediction of registered model

  Registered models : ['graphcast', 'pangu', 'sfno', 'fuxi']

  All other keyword arguments will be passed to underlying model

  MODEL: Model to use

Options:
  --time TEXT             Time to predict from  [required]
  --pipeline TEXT         Pipeline config  [required]
  --output DIRECTORY      Output directory  [required]
  --data_cache DIRECTORY  Data cache location
  --debug                 Debug Mode
  --help                  Show this message and exit.
```

## Interactive

```shell
[~]$ edit-models interactive --help

Usage: edit-models interactive [OPTIONS]

  Interactive Prediction

  Run Prediction of registered model, getting arguments interactively

  Registered models : ['graphcast', 'pangu', 'sfno', 'fuxi']

  If a keyword argument is not given, this command will prompt the user,
  however, if passed, will skip.

  MODEL: Model to use

Options:
  --debug  Debug mode
  --help   Show this message and exit.

```

## Models

```shell
[~]$ edit-models models
Available models:
        graphcast
        sfno
        pangu
        fuxi
```

## Data

```shell
[~]$ edit-models data

Usage: edit-models data [OPTIONS] MODEL

  Data retrieval

  Get data for registered model

  Registered models : ['graphcast', 'pangu', 'sfno', 'fuxi']

  MODEL: Model to get data for

Options:
  --time TEXT             Basetime to get data for  [required]
  --pipeline TEXT         Pipeline config  [required]
  --data_cache DIRECTORY  Data cache location
  --debug                 Debug mode
  --help                  Show this message and exit.
```

## Optional Arguments

As the `kwargs` will get passed to the `edit.training.MLDataIndex`, any kwargs that accepts, the commands will too.

The following table lists some useful ones.

| Name | Purpose | Type |
| ---- | ------- | ---- |
| overrride | Override existing data | `bool` |
| cleanup | Clean up cache config | `str / int` (Either Directory size or file time limit in days) |
