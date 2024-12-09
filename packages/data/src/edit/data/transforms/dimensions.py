# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations
from typing import Any, Optional, Literal, TypeVar

import xarray as xr

from pyearthtools.data.transforms.transform import Transform
from pyearthtools.utils.decorators import BackwardsCompatibility

XR = TypeVar("XR", xr.Dataset, xr.DataArray)


class StandardDimensionNames(Transform):
    """Standardise dimension names"""

    def __init__(self, replacement_dictionary: Optional[dict[str, str]] = None, **kwargs: str):
        """
        Convert Dataset Dimension Names into Standard Naming Scheme

        Args:
            replacement_dictionary (dict[Hashable, Hashable]):
                Dictionary assigning dimension name replacements [old: new]

        Returns:
            (Transform):
                Transform to replace dimension names
        """
        super().__init__()
        self.record_initialisation()

        replacement_dictionary = replacement_dictionary or {}
        replacement_dictionary.update(kwargs)

        self._replacement_dictionary = replacement_dictionary

    @property
    def _info_(self):
        return dict(**self._replacement_dictionary)

    def apply(self, dataset: xr.Dataset):
        for correctname, falsenames in self._replacement_dictionary.items():
            for falsename in set(falsenames) & (set(dataset.dims).union(set(dataset.coords))):
                dataset = dataset.rename({falsename: correctname})
                if falsename in dataset:
                    dataset = dataset.drop(falsename)
        return dataset


@BackwardsCompatibility(StandardDimensionNames)
def force_standard_dimension_names(*args, **kwargs: str) -> Transform: ...


class Expand(Transform):
    """Expand Dimensions"""

    def __init__(
        self,
        dim: list[str] | dict[str, int] | str | None = None,
        axis: int | list[int] | None = None,
        as_dataarray: bool = True,
        missing: Literal["skip", "error"] = "error",
        exists: Literal["skip", "error"] = "error",
        **kwargs: int,
    ):
        """
        Expand Dimensions.

        Uses `xarray` `.expand_dims`.

        Args:
            dim (list[str] | dict | str | None, optional):
                Dimensions to include on the new variable.
                If provided as str or sequence of str, then dimensions are inserted with length 1.
                If provided as a dict, then the keys are the new dimensions and the values are either integers
                (giving the length of the new dimensions) or sequence/ndarray (giving the coordinates of the new dimensions).
            axis (int | list[int] | None, optional):
                Axis position(s) where new axis is to be inserted (position(s) on the result array).
                If a sequence of integers is passed, multiple axes are inserted. In this case, dim arguments should be same length list.
                If axis=None is passed, all the axes will be inserted to the start of the result array.
            as_dataarray (bool, optional):
                Expand each variable independently. Defaults to True.
            missing (Literal['skip','error'], optional):
                What to do when a missing `dim` is given. Defaults to 'error'.
            kwargs (int):
                Keywords form of `dim`.
        """
        super().__init__()
        self.record_initialisation()

        self._dim = dim
        self._axis = axis
        self._as_dataarray = as_dataarray

        self._missing = missing
        self._exists = exists

        self._kwargs = kwargs

    def apply(self, dataset: XR) -> XR:

        if self._as_dataarray and isinstance(dataset, xr.Dataset):
            for var in dataset.data_vars:
                dataset[var] = self.apply(dataset[var])  # .expand_dims(_dim, axis=self._axis, **self._kwargs)
            return dataset

        _dim = self._dim

        if self._missing == "skip":
            if isinstance(_dim, str) and _dim not in dataset.coords:
                _dim = None
            if isinstance(_dim, list):
                _dim = list(set(_dim).intersection(dataset.coords))

        kwargs = dict(self._kwargs)
        if self._exists == "skip":
            for key in self._kwargs.keys():
                if key in dataset.dims:
                    kwargs.pop(key)

            if isinstance(_dim, str) and _dim in dataset.dims:
                _dim = None
            if isinstance(_dim, list):
                _dim = list(set(_dim).difference(dataset.dims))

        data = dataset.expand_dims(_dim, axis=self._axis, **self._kwargs)
        return data


@BackwardsCompatibility(Expand)
def expand(*args, **kwargs) -> Transform: ...
