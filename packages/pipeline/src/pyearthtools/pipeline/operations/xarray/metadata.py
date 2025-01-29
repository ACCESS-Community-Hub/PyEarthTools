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


from typing import TypeVar, Optional, Any, Literal

import xarray as xr

import pyearthtools.data

from pyearthtools.pipeline.operation import Operation

T = TypeVar("T", xr.Dataset, xr.DataArray)


class Rename(Operation):
    """
    Rename `variables` in an `xr.Dataset`.
    """

    _override_interface = "Serial"

    def __init__(self, rename: Optional[dict[str, str]], **rename_kwargs):
        """
        Rename variables in an `xr.Dataset`

        Args:
            rename (Optional[dict[str, str]]):
                Name conversion dictionary
        """

        super().__init__(
            split_tuples=True,
            recursively_split_tuples=True,
            recognised_types=(xr.Dataset,),
        )
        self.record_initialisation()
        rename = rename or {}
        rename.update(rename_kwargs)

        self._rename = rename

    def apply_func(self, sample: xr.Dataset) -> xr.Dataset:
        return pyearthtools.data.transforms.attributes.rename(self._rename)(sample)

    def undo_func(self, sample: xr.Dataset) -> xr.Dataset:
        return pyearthtools.data.transforms.attributes.rename({val: key for key, val in self._rename.items()})(sample)


class Encoding(Operation):
    """
    Set encoding on `xarray` objects
    """

    _override_interface = "Serial"

    def __init__(
        self,
        encoding: dict[str, dict[str, Any]],
        operation: Literal["apply", "undo", "both"] = "both",
    ):
        """
        Set encoding on `xarray` objects

        Args:
            encoding (dict[str, dict[str, Any]]):
                Variable value pairs assigning encoding to the given variable.
                Can set key to 'all' to apply to all variables.
                Defaults to None.
            operation (Literal['apply', 'undo', 'both'], optional):
                When to apply encoding setting. Defaults to "both".
        """
        super().__init__(
            split_tuples=True,
            recursively_split_tuples=True,
            operation=operation,
            recognised_types=(xr.Dataset, xr.DataArray),
        )
        self.record_initialisation()
        self._encoding = pyearthtools.data.transforms.attributes.set_encoding(encoding)

    def apply_func(self, sample: T) -> T:
        return self._encoding(sample)

    def undo_func(self, sample: T) -> T:
        return self._encoding(sample)


class MaintainEncoding(Operation):
    _encoding: Optional[pyearthtools.data.Transform] = None
    _override_interface = "Serial"

    def __init__(self, reference: Optional[str] = None, limit: Optional[list[str]] = None):
        """
        Maintain encoding of samples from `apply` to `undo`.

        If `apply` not called before `undo`, this will do nothing.

        Args:
            reference Optional[str], optional):
                Reference dataset to get encoding from. If not given will use first `sample` on `apply`.
            limit (Optional[list[str]], optional):
                When getting encoding from `reference` object, limit the retrieved encoding.
                If not given will get `['units', 'dtype', 'calendar', '_FillValue', 'scale_factor', 'add_offset', 'missing_value']`.
                Defaults to None.
        """
        super().__init__(
            split_tuples=True,
            recursively_split_tuples=True,
            recognised_types=(xr.Dataset, xr.DataArray),
        )
        self.record_initialisation()
        self._encoding = (
            None
            if reference is None
            else pyearthtools.data.transforms.attributes.set_encoding(reference=xr.open_dataset(reference), limit=limit)
        )
        self._limit = limit

    def apply_func(self, sample: T) -> T:
        if not self._encoding:
            self._encoding = pyearthtools.data.transforms.attributes.set_encoding(reference=sample, limit=self._limit)
        return sample

    def undo_func(self, sample: T) -> T:
        if not self._encoding:
            return sample
        return self._encoding(sample)


class Attributes(Operation):
    """
    Set attributes on `xarray` objects
    """

    _override_interface = "Serial"

    def __init__(
        self,
        attributes: dict[str, dict[str, Any]],
        apply_on: Literal["dataset", "dataarray", "both", "per_variable"] = "dataset",
        operation: Literal["apply", "undo", "both"] = "both",
    ):
        """
        Set attributes on `xarray` objects

        Args:
            attrs (dict[str, Any] | None):
                Attributes to set, key: value pairs.
                Set `apply_on` to choose where attributes are applied.
                | Key | Description |
                | --- | ----------- |
                | dataset | Attributes updated on dataset |
                | dataarray | If applied on a dataset, update each dataarray inside the dataset |
                | both | Do both above |
                | per_variable | Treat `attrs` as a dictionary of dictionaries, applying on dataarray if in dataset. |
                Defaults to None.
            apply_on (Literal['dataset', 'dataarray', 'both'], optional):
                On what type to update attributes. Defaults to 'dataset'.
            operation (Literal['apply', 'undo', 'both'], optional):
                When to apply encoding setting. Defaults to "both".
        """
        super().__init__(
            split_tuples=True,
            recursively_split_tuples=True,
            operation=operation,
            recognised_types=(xr.Dataset, xr.DataArray),
        )
        self.record_initialisation()
        self._attributes = pyearthtools.data.transforms.attributes.set_attributes(attrs=attributes, apply_on=apply_on)

    def apply_func(self, sample: T) -> T:
        return self._attributes(sample)

    def undo_func(self, sample: T) -> T:
        return self._attributes(sample)


class MaintainAttributes(Operation):
    """
    Maintain attributes
    """

    _attributes: Optional[pyearthtools.data.Transform] = None
    _override_interface = "Serial"

    def __init__(self, reference: Optional[str] = None):
        """
        Maintain attributes of samples from `apply` to `undo`.

        If `apply` not called before `undo`, this will do nothing.

        Args:
            reference Optional[str], optional):
                Reference dataset to get attributes from. If not given will use first `sample` on `apply`.
        """
        super().__init__(
            split_tuples=True,
            recursively_split_tuples=True,
            recognised_types=(xr.Dataset, xr.DataArray),
        )
        self.record_initialisation()
        self._attributes = (
            None
            if reference is None
            else pyearthtools.data.transforms.attributes.set_attributes(reference=xr.open_dataset(reference))
        )

    def apply_func(self, sample: T) -> T:
        if not self._attributes:
            self._attributes = pyearthtools.data.transforms.attributes.set_attributes(reference=sample)
        return sample

    def undo_func(self, sample: T) -> T:
        if not self._attributes:
            return sample
        return self._attributes(sample)
