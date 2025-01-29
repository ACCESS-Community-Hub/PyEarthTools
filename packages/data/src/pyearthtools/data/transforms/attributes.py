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
Attribute modification
"""

from __future__ import annotations
from typing import Any, Literal

import xarray as xr

from pyearthtools.data.transforms import Transform
from pyearthtools.utils.decorators import BackwardsCompatibility


def _get_attributes_from_ds(reference: xr.DataArray | xr.Dataset) -> dict[str, dict[str, Any]]:
    attributes: dict[str, dict[str, Any]] = {}

    if isinstance(reference, xr.DataArray):
        reference = reference.to_dataset()

    for var in (*reference.dims, *reference.data_vars):
        if var not in attributes:
            attributes[str(var)] = {}
        for rel in reference[var].attrs:
            attributes[str(var)][rel] = reference[var].attrs[rel]

    return attributes


class SetAttributes(Transform):
    """Set Attributes"""

    def __init__(
        self,
        attrs: dict[str, Any] | None = None,
        reference: xr.DataArray | xr.Dataset | None = None,
        apply_on: Literal["dataset", "dataarray", "both", "per_variable"] = "dataset",
        **attributes,
    ):
        """
        Modify Attributes to a dataset

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
            **attributes (dict):
                Keyword argument form of `attrs`.

        Returns:
            (Transform):
                Transform to set attributes
        """
        super().__init__()
        self.record_initialisation()

        _attributes = {}
        if reference is not None:
            apply_on = "per_variable"
            _attributes.update(_get_attributes_from_ds(reference))

        _attributes.update(attrs or {})
        _attributes.update(**attributes)
        self._attributes = _attributes
        self._apply_on = apply_on

    # @property
    # def _info_(self):
    #     return dict(**self._attributes, apply_on=self._apply_on)

    def apply(self, data_obj):
        if self._apply_on in ["both", "dataarray"] and isinstance(data_obj, xr.Dataset):
            for var in data_obj.data_vars:
                data_obj[var] = self.apply(data_obj[var])

            if self._apply_on == "dataarray":
                return data_obj

        for key, value in self._attributes.items():
            if self._apply_on == "per_variable":
                if key in data_obj and isinstance(data_obj, xr.Dataset):
                    data_obj[key].attrs.update(**value)
            else:
                data_obj.attrs.update(**{key: value})
        return data_obj


@BackwardsCompatibility(SetAttributes)
def set_attributes(*args, **kwargs) -> Transform: ...


update = set_attributes


def _get_encoding_from_ds(reference: xr.DataArray | xr.Dataset, limit: list[str] | None = None):
    encoding: dict[str, dict[str, Any]] = {}
    relevant_encoding = limit or [
        "units",
        "dtype",
        "calendar",
        "_FillValue",
        "scale_factor",
        "add_offset",
        "missing_value",
    ]

    if isinstance(reference, xr.DataArray):
        reference = reference.to_dataset()

    for var in (*reference.dims, *reference.data_vars):
        if var not in encoding:
            encoding[var] = {}
        for rel in set(relevant_encoding).intersection(set(list(reference[var].encoding.keys()))):
            encoding[var][rel] = reference[var].encoding[rel]
    return encoding


class SetEncoding(Transform):
    """Set Encoding"""

    def __init__(
        self,
        encoding: dict[str, dict[str, Any]] | None = None,
        reference: xr.DataArray | xr.Dataset | None = None,
        limit: list[str] | None = None,
        **variables,
    ):
        """
        Set encoding of a dataset.

        Can get encoding from a reference dataset. That dataset is then not used, as the encoding has already been retrieved.

        Args:
            encoding (dict[str, dict[str, Any]] | None):
                Variable value pairs assigning encoding to the given variable.
                Can set key to 'all' to apply to all variables.
                Defaults to None.
            reference (xr.DataArray | xr.Dataset | None, optional):
                Reference object to retrieve and update encoding from. Defaults to None.
            limit (list[str] | None, optional):
                When getting encoding from `reference` object, limit the retrieved encoding.
                If not given will get `['units', 'dtype', 'calendar', '_FillValue', 'scale_factor', 'add_offset', 'missing_value']`.
                Defaults to None.
            **variables (dict):
                Keyword argument form of `encoding`
        """
        super().__init__()
        self.record_initialisation()

        if encoding is None:
            encoding = {}
        encoding = dict(encoding)
        encoding.update(**dict(variables))

        if reference is not None:
            encoding.update(**_get_encoding_from_ds(reference, limit=limit))

        self._encoding = encoding

    # @property
    # def _info_(self):
    #     return dict(**self._encoding)

    def apply(self, dataset: xr.Dataset) -> xr.Dataset:
        if isinstance(dataset, xr.DataArray):
            new_ds = self.apply(dataset.to_dataset(name=dataset.name or "data"))
            return new_ds[list(new_ds.data_vars)[0]]

        for key, value in self._encoding.items():
            if key == "all":

                def update(x: xr.DataArray):
                    x.encoding.update(**value)
                    return x

                dataset = dataset.map(update)
                continue

            if key in dataset:
                dataset[key].encoding.update(**value)
        return dataset


@BackwardsCompatibility(SetEncoding)
def set_encoding(*args, **kwargs) -> Transform: ...


class SetType(Transform):
    """Set type of variables"""

    def __init__(self, dtype: str | dict[str, str] | None = None, **variables: Any):
        """
        Set type of variables/coordinates.

        At least `dtype` or `variables` must be set.

        Applies "same_kind" casting

        Args:
            dtype (str | dict[str, str] | None):
                Datatype to set to. If only `dtype` is given,
                this will set all coordinates of the dataset to this `dtype`.
                Defaults to None.
            **variables (Any, optional):
                Variable dtype configuration.
        """
        super().__init__()
        self.record_initialisation()

        if not dtype and not variables:
            raise ValueError(f"Either `dtype` or `**variables` must be given.")

        self._dtype = dtype
        self._variables = variables

    # @property
    # def _info_(self):
    #     return dict(dtype=self._dtype, **self._variables)

    def apply(self, dataset: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
        if not isinstance(self._dtype, dict):
            self._variables.update({str(coord): self._dtype for coord in dataset.coords})

        for var, dt in self._variables.items():
            try:
                dataset[var] = dataset[var].astype(dt, casting="same_kind")
            except TypeError:
                pass
        return dataset


@BackwardsCompatibility(SetType)
def set_type(*args, **kwargs) -> Transform: ...


class Rename(Transform):
    """Rename Components inside Dataset"""

    def __init__(self, names: dict[str, Any] | None = None, **extra_names: Any):
        """
        Rename Dataset components

        Args:
            names (dict[str, Any] | None):
                Dictionary assigning name replacements [old: new]
                Defaults to None.
            **extra_names (Any, optional):
                Keyword args form of `names`.
        """
        super().__init__()
        self.record_initialisation()

        names = names or {}

        names = dict(names)
        names.update(extra_names)
        self._names = names

    # @property
    # def _info_(self):
    #     return dict(**self._names)

    def apply(self, dataset: xr.Dataset) -> xr.Dataset:
        return dataset.rename(**{key: self._names[key] for key in self._names if key in dataset})


@BackwardsCompatibility(Rename)
def rename(*args, **kwargs) -> Transform: ...
