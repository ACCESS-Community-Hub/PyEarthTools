# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""

Extend functionality of `pyearthtools.data.indexes`.

Largely sourced from [xarray.extensions](https://docs.xarray.dev/en/stable/internals/extending-xarray.html)

[GitHub Code](https://github.com/pydata/xarray/blob/main/xarray/core/extensions.py)

Here is how `pyearthtools.plotting.geo` in effect extends the `indexers`
```python
@pyearthtools.data.register_accessor("geo", 'DataIndex')
class GeoAccessor:
    def __init__(self, pyearthtools_obj):
        self._obj = pyearthtools_obj

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

import pyearthtools.data
from pyearthtools.data.indexes.indexes import Index


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
                pyearthtools.data.AccessorRegistrationWarning,
                stacklevel=2,
            )
        setattr(cls, name, _CachedAccessor(name, accessor))
        return accessor

    return decorator


def register_accessor(name: str, object: str | type | ModuleType = Index) -> Callable:
    """
    Register a custom accessor on `pyearthtools.data` indexes.

    Any decorated class will receive the `pyearthtools.data.Index` as it's first and only argument.

    Args:
        name (str):
            Name under which the accessor should be registered. A warning is issued
            if this name conflicts with a preexisting attribute.
        object (str | type | ModuleType, optional):
            `pyearthtools.data.indexes` object to register accessor to.
            By default this will add to the base level index, so is available from all.
            Defaults to Index.

    Examples:
        In your library code:

        >>> @pyearthtools.data.register_accessor("geo", 'DataIndex')
        ... class GeoAccessor:
        ...     def __init__(self, pyearthtools_obj):
        ...         self._obj = pyearthtools_obj

        ...     # Using the `pyearthtools.data.Index`, retrieve data and do something.
        ...     def plot(self):
        ...         # Run plotting
        ...         pass
        ...

        Back in an interactive IPython session:

        >>> era5 = pyearthtools.data.archive.ERA5(
        ...     variables = '2t', level = 'single'
        ... )
        >>> era5.geo.plot()  # plots index on a map
    """

    if isinstance(object, str):
        if not hasattr(pyearthtools.data, object):
            raise ValueError(f"Cannot find {object!r} underneath `pyearthtools.data`.")
        object = getattr(pyearthtools.data, object)
    assert not isinstance(object, str)
    return _register_accessor(name, object)
