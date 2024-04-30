# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

"""
Attribute modification
"""

from __future__ import annotations
from typing import Any, Literal

import xarray as xr

from edit.data.transform import Transform


def set_attributes(
    attrs: dict[str, Any] | None = None,
    apply_on: Literal["dataset", "dataarray", "both", "per_variable"] = "dataset",
    **attributes,
) -> Transform:
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

    if attrs is None:
        attrs = {}
    attrs = dict(attrs)
    attrs.update(**attributes)

    class ModifyAttributes(Transform):
        """Modify Attributes"""

        @property
        def _info_(self):
            return dict(**attrs, apply_on=apply_on)

        def apply(self, data_obj: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
            if apply_on in ["both", "dataarray"] and isinstance(data_obj, xr.Dataset):
                for var in data_obj.data_vars:
                    data_obj[var] = self.apply(data_obj[var])

                if apply_on == "dataarray":
                    return data_obj

            for key, value in attrs.items():
                if apply_on == "per_variable":
                    if key in data_obj and isinstance(data_obj, xr.Dataset):
                        data_obj[key].attrs.update(**value)
                else:
                    data_obj.attrs.update(**{key: value})
            return data_obj

    return ModifyAttributes()


update = set_attributes


def _get_encoding_from_ds(reference: xr.DataArray | xr.Dataset, limit: list[str] | None = None):
    encoding = {}
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


def set_encoding(
    encoding: dict[str, dict] | None = None,
    reference: xr.DataArray | xr.Dataset | None = None,
    limit: list[str] | None = None,
    **variables,
) -> Transform:
    """
    Set encoding of a dataset.

    Can get encoding from a reference dataset. That dataset is then not used, as the encoding has already been retrieved.

    Args:
        encoding (dict[str, dict] | None):
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

    Returns:
        (Transform):
            Transform to add attrs
    """
    if encoding is None:
        encoding = {}
    encoding = dict(encoding)
    encoding.update(**dict(variables))

    if reference is not None:
        encoding.update(**_get_encoding_from_ds(reference, limit=limit))

    class SetEncoding(Transform):
        """Set Encoding"""

        @property
        def _info_(self):
            return dict(**encoding)

        def apply(self, dataset: xr.Dataset) -> xr.Dataset:
            if isinstance(dataset, xr.DataArray):
                new_ds = self.apply(dataset.to_dataset())
                return new_ds[list(new_ds.data_vars)[0]]

            for key, value in encoding.items():
                if key == "all":

                    def update(x: xr.DataArray):
                        x.encoding.update(**value)
                        return x

                    dataset = dataset.map(update)
                    continue

                if key in dataset:
                    dataset[key].encoding.update(**value)
            return dataset

    return SetEncoding()


def set_type(dtype: str | dict[str, str] | None = None, **variables) -> Transform:
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

    Returns:
        (Transform):
            Transform to set datatypes
    """

    if not dtype and not variables:
        raise ValueError(f"Either `dtype` or `**variables` must be given.")

    class SetType(Transform):
        """Set variables to consistent type. Skip if TypeError"""

        @property
        def _info_(self):
            return dict(dtype=dtype)

        def apply(self, dataset: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
            if not isinstance(dtype, dict):
                variables.update({coord: dtype for coord in dataset.coords})

            for var, dt in variables.items():
                try:
                    dataset[var] = dataset[var].astype(dt, casting="same_kind")
                except TypeError:
                    pass
            return dataset

    return SetType()


def rename(names: dict[str, Any] | None = None, **extra_names: Any) -> Transform:
    """
    Rename Dataset components

    Args:
        names (dict[str, Any] | None):
            Dictionary assigning name replacements [old: new]
            Defaults to None.
        **extra_names (Any, optional):
            Keyword args form of `names`.

    Returns:
        Transform: Transform to apply name replacements
    """
    if names is None:
        names = {}

    names = dict(names)
    names.update(extra_names)

    class Rename(Transform):
        """Rename variables in Dataset"""

        @property
        def _info_(self):
            return dict(**names)

        def apply(self, dataset: xr.Dataset) -> xr.Dataset:
            return dataset.rename(**{key: names[key] for key in names if key in dataset})

    return Rename()
