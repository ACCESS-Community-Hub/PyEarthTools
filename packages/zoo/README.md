# `pyearthtools.zoo`

Allows for the configuration and registration of models and then the creation of an easy to use CMD to run forecasts.

Using `pyearthtools` data and pipelines, a model can be setup to pull from any number of sources.

## Implemented

As of 05-2025, the following models are available.

```mermaid
graph TD
    Next[FourCastNeXt]
```

## Usage

The `pyearthtools` CLI can be used both in the command line and within a python environment.

### Command line

```bash
# To run a prediction using a local copy of ERA5

[user@potato ~]$ pet predict next --pipeline ERA5 --output /forecasts/next/ --lead_time '14-days' --time 2024-01-06T03
```

### Programtically

`pyearthtools.zoo` provides the following top level functions for easy usage in a python environment.

#### `.predict`

Run predictions.

All arguments, such as the model, pipeline and directories must be fully specified.

This function will then import the model, download the assets if needed, and then run the prediction.

```python
import pyearthtools.zoo as zoo

predictions = zoo.predict('model_name_here', 'pipeline_name_here', *args, **kwargs)
```

## Advanced Usage

### Changing Output File Pattern Structure

By default, all data output from a model is saved in a variable aware, date expanded form, (`ForecastExpandedDateVariable`). This was chosen as the default as it closely matched many existing data archive structures.

However, as this uses the `pyearthtools` patterns to allow this, it is quite easy to adjust this and change the structure the data is saved in.

Any pattern listed in `pyearthtools.data` can be used, by providing it's class name to the `pyearthtools-zoo` call.

```shell

pet predict next OTHER_ARGS --pattern ExpandedDate
```

This will now use the `ExpandedDate` pattern, saving data in this case at:

```txt
temp_dir/2023/01/01/20230101T0000.nc
```

### Adding Arguments for Patterns

These patterns can take extra keyword arguments to further control the behaviour and layout of the saved data.

To specify these kwargs, add `--pattern_kwargs` and provide a dictionary in a json form, .i.e.

```shell
pyearthtools-models predict next OTHER_ARGS --pattern ExpandedDate --pattern_kwargs '{"directory_resolution":"month", "prefix":"_test_"}
```

This will now use the `ExpandedDate` pattern, saving data in this case at:

```txt
temp_dir/2023/01/_test_20230101T0000.nc
```
