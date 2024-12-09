# Transforms

Using `pyearthtools.data.DataIndex`'s which most archives subclass from, allows the use of `Transforms`.

## Definition

A `Transform` is an atomic operation to directly modify the data. 
Examples include,

- Variable Subsetting
- Dimension Reordering
- Variable Derivation
- Interpolation
- ...

These can be easily added to various parts of the data loading pipeline to directly modify the data, or can be used outside of `pyearthtools.data.Indexes`, and directly used on xarray object.s

Ultimately, a `Transform` operates on a given data object seperate to its source. It introduces no new information, but simply 'transforms' the information already contained.

For a more complex `Transform` like operation, checkout [Modifications](./Modifications.md).

## Example

Below is the actual `Transform` for variable trimming, (`pyearthtools.data.transforms.variables.Trim`).

The root `Transform` class handles the extra neccessary functionality for class manipulation, calling and combining of multiple transforms. A child `Transform` needs only implement `apply`. `apply` is the actual functionality and will be given a `dataset` object, and must return the same type of object.

```python
class Trim(Transform):
    """Trim dataset variables"""

    def __init__(self, variables: list[str] | str, *extra_variables):
        """
        Trim Dataset to given variables.

        If no variables would be left, apply no Transform

        Args:
            variables (list[str] | str):
                List of vars to trim to
        """
        super().__init__()
        self.record_initialisation()

        variables = variables if isinstance(variables, (list, tuple)) else [variables]
        self._variables = [*variables, *extra_variables]

    def apply(self, dataset: xr.Dataset) -> xr.Dataset:
        if self._variables is None:
            return dataset
        var_included = set(self._variables) & set(dataset.data_vars)
        if not var_included:
            return dataset
        return dataset[var_included]

```

### Repr & Saving Information

`self.record_initialisation()` is used to record the initilisation arguments and allow for saving to a yaml file.

It must be called after `super().__init__()`.