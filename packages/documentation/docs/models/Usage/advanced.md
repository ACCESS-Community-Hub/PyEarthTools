# Altering Patterns

## Changing Output File Pattern Structure

By default, all data output from a model is saved in a variable aware, date expanded form, (`ForecastExpandedDateVariable`). This was chosen as the default as it closely matched many existing data archive structures.

However, as this uses the `pyearthtools` patterns to allow this, it is quite easy to adjust this and change the structure the data is saved in.

Any pattern listed [here][pyearthtools.data.patterns] can be used, by providing it's class name to the `pyearthtools-models` call.

!!! tip "Pattern Specification"
    ```shell

    pyearthtools-models predict graphcast OTHER_ARGS --pattern ExpandedDate

    ```
    This will now use the `ExpandedDate` pattern, saving data in this case at 
    ```txt
    temp_dir/2023/01/01/20230101T0000.nc
    ```

### Adding Arguments for Patterns

These patterns can take extra keyword arguments to further control the behaviour and layout of the saved data,

To specify these kwargs, add `--pattern_kwargs` and provide a dictionary in a json form, .i.e.

!!! tip "Pattern Kwargs"
    ```shell

    pyearthtools-models predict graphcast OTHER_ARGS --pattern ExpandedDate --pattern_kwargs '{"directory_resolution":"month", "prefix":"_test_"}

    ```
    This will now use the `ExpandedDate` pattern, saving data in this case at 
    ```txt
    temp_dir/2023/01/_test_20230101T0000.nc
    ```
