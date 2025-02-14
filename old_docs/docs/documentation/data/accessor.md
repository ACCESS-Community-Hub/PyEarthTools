# Extending `pyearthtools`

`pyearthtools` provides a method to add features to the [indexes][pyearthtools.data.Index].

This is modelled after the [xarray accessors](https://docs.xarray.dev/en/stable/internals/extending-xarray.html), and allows a user to add methods and functionality automatically to any `index`.

In addition, the extension can be registered to any specific `index`, and thus use limited to time based indexes or the `ArchiveIndex`.

In your library code:

```py
@pyearthtools.data.register_accessor("geo", 'DataIndex')
class GeoAccessor:
    def __init__(self, pyearthtools_obj : pyearthtools.data.DataIndex):
        self._obj = pyearthtools_obj
    def plot(self):
        # plot this index's data on a map, e.g., using Cartopy
        pass
```

Back in an interactive IPython session:

```py

era5 = pyearthtools.data.archive.ERA5(
    variables = '2t', level = 'single'
)
era5.geo.plot()  # plots index on a map
```

In general, the only restriction on the accessor class is that the `__init__` method must have a single parameter: the `Index` it is supposed to work on.

This `register_accessor` will be cached onto the object, and the name checked to ensure it does not accidently confict with any other attributes or methods (including other accessors).
