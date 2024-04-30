# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

"""

Extend functionality of `edit.data.indexes`.

Largely sourced from [xarray.extensions](https://docs.xarray.dev/en/stable/internals/extending-xarray.html)

[GitHub Code](https://github.com/pydata/xarray/blob/main/xarray/core/extensions.py)

Here is how `edit.plotting.geo` in effect extends the `indexers`
```python
@edit.data.register_accessor("geo", 'DataIndex')
class GeoAccessor:
    def __init__(self, edit_obj):
        self._obj = edit_obj

    def plot(self):
        # plot this index's data on a map, e.g., using Cartopy
        pass

```
In general, the only restriction on the accessor class is that the `__init__` method must have a single parameter: the `Index` object it is supposed to work on.

This achieves the same result as if the `Index` class had a cached property defined that returns an instance of the class:
```python
class Index:
    ...

    @property
    def geo(self):
        return GeoAccessor(self)

```
"""

from __future__ import annotations

import warnings
from types import ModuleType
from typing import Callable

import edit.data
from edit.data.indexes.indexes import Index


class _CachedAccessor:
    """Custom property-like object (descriptor) for caching accessors."""

    def __init__(self, name: str, accessor: Callable):
        self._name = name
        self._accessor = accessor

    def __get__(self, obj, cls):
        if obj is None:
            # we're accessing the attribute of the class, i.e., Index.geo
            return self._accessor

        try:
            cache = obj._cache
        except AttributeError:
            cache = obj._cache = {}

        try:
            return cache[self._name]
        except KeyError:
            pass

        try:
            accessor_obj = self._accessor(obj)
        except AttributeError:
            # __getattr__ on data object will swallow any AttributeErrors
            # raised when initializing the accessor, so we need to raise as
            # something else (GH933):
            raise RuntimeError(f"error initializing {self._name!r} accessor.")

        cache[self._name] = accessor_obj
        return accessor_obj


def _register_accessor(name: str, cls: ModuleType | type) -> Callable:
    def decorator(accessor):
        if hasattr(cls, name):
            warnings.warn(
                f"Registration of accessor {accessor!r} under name {name!r} for type {cls!r} is "
                "overriding a preexisting attribute with the same name.",
                edit.data.AccessorRegistrationWarning,
                stacklevel=2,
            )
        setattr(cls, name, _CachedAccessor(name, accessor))
        return accessor

    return decorator


def register_accessor(name: str, object: str | type | ModuleType = Index) -> Callable:
    """
    Register a custom accessor on `edit.data` indexes.

    Any decorated class will receive the `edit.data.Index` as it's first and only argument.

    Args:
        name (str):
            Name under which the accessor should be registered. A warning is issued
            if this name conflicts with a preexisting attribute.
        object (str | type | ModuleType, optional):
            `edit.data.indexes` object to register accessor to.
            By default this will add to the base level index, so is available from all.
            Defaults to Index.

    Examples:
        In your library code:

        >>> @edit.data.register_accessor("geo", 'DataIndex')
        ... class GeoAccessor:
        ...     def __init__(self, edit_obj):
        ...         self._obj = edit_obj

        ...     # Using the `edit.data.Index`, retrieve data and do something.
        ...     def plot(self):
        ...         # Run plotting
        ...         pass
        ...

        Back in an interactive IPython session:

        >>> era5 = edit.data.archive.ERA5(
        ...     variables = '2t', level = 'single'
        ... )
        >>> era5.geo.plot()  # plots index on a map
    """

    if isinstance(object, str):
        if not hasattr(edit.data, object):
            raise ValueError(f"Cannot find {object!r} underneath `edit.data`.")
        object = getattr(edit.data, object)
    assert not isinstance(object, str)
    return _register_accessor(name, object)
