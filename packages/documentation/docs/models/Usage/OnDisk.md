# On Disk Modifications

Using `edit-models` experiments can be run with altered data, i.e. with different initial conditions.

This takes advantage of the `edit-models data` subcommand to save original data, and assumes the user modifies this data, and then the configurable config location to use the doctored initial conditions.

## Getting Initial Conditions

Use `edit-models data` to save the initial conditions for any model in use.

i.e. for `graphcast`

```shell
edit-models data 'Global/graphcast' --time 2020-01-01T00 --pipeline 'ERA5' --data_cache $CACHE_DIRECTORY $OTHER_KWARGS
```

It may be neccessary to add extra kwargs depending on the model, .i.e `lead_time`. If it seems irrelevant for the getting of data, give a dummy value.

Once the first bit of data is saved, it would be possible to load the saved `pipeline.yaml`, and use independently of `edit-models` to save more initial conditions.

If doing so, alter the `dir_size` value within the `CachingIndex` to allow more data to be saved.

## Setting a config

Add the root model config directory to the environmental variable

```shell
prepend_path EDIT_MODELS_CONFIGS $CONFIG_DIRECTORY
```

### Writing the config

Write the config underneath the full model categorical path, + `Data/`
i.e.

```shell
    $CONFIG_DIRECTORY/Global/graphcast/Data/$CONFIG_NAME.yaml
```

Then set the contents of that yaml to be like the following, changing only the location of the data in `edit.data.load`.

```yaml
edit.data.load:
  - $CACHE_DIRECTORY/$MODEL/$PIPELINE/catalog.cat
edit.pipeline.indexes.cache.CachingIndex:
  cache: 'temp'
```

## Running the Forecast

Below shows an example of running a forecast, with the model being selected as `graphcast` and the config as named given.

```shell
edit-models predict 'graphcast' --pipeline $CONFIG_NAME --output $DATA_SAVE_DIRECTORY --time '2020-01-01T00' --lead_time '14-days'
```
