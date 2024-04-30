# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

from __future__ import annotations
from typing import Any, Hashable, Literal

import xarray as xr

from edit.data.transform.transform import Transform


def force_standard_dimension_names(replacement_dictionary: dict[str, str], **kwargs: str) -> Transform:
    """
    Convert Dataset Dimension Names into Standard Naming Scheme

    Args:
        replacement_dictionary (dict[Hashable, Hashable]):
            Dictionary assigning dimension name replacements [old: new]

    Returns:
        (Transform):
            Transform to replace dimension names
    """
    replacement_dictionary.update(kwargs)

    class ConformNaming(Transform):
        """Force Standard Dimension Names"""

        @property
        def _info_(self):
            return dict(**replacement_dictionary)

        def apply(self, dataset: xr.Dataset):
            for correctname, falsenames in replacement_dictionary.items():
                for falsename in set(falsenames) & (set(dataset.dims) + set(dataset.coords)):
                    dataset = dataset.rename({falsename: correctname})
                    if falsename in dataset:
                        dataset = dataset.drop(falsename)
            return dataset

    return ConformNaming()


def expand(
    dim: list[str] | dict[str, int] | str | None = None,
    axis: int | list[int] | None = None,
    as_dataarray: bool = True,
    missing: Literal["skip", "error"] = "error",
    **kwargs: int,
) -> Transform:
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
    Returns:
        (Transform):
            Transform to expand dims
    """

    class ExpandDims(Transform):
        """
        Expand dimensions of data
        """

        def apply(self, dataset: xr.Dataset) -> xr.Dataset:
            _dim = dim
            if missing == "skip":
                if isinstance(_dim, str) and _dim not in dataset.coords:
                    _dim = None
                if isinstance(_dim, list):
                    _dim = list(set(_dim).intersection(dataset.coords))

            if as_dataarray:
                for var in dataset.data_vars:
                    dataset[var] = dataset[var].expand_dims(_dim, axis=axis, **kwargs)
                return dataset
            data = dataset.expand_dims(**kwargs)
            return data

        @property
        def _info_(self) -> Any | dict:
            return dict(dim=dim, axis=axis, as_dataarray=as_dataarray, missing=missing, **kwargs)

    return ExpandDims()


# def order(dim: list[str] | tuple[str] | str, *dims):
#     dim = [dim] if not isinstance(dim, (list, tuple)) else dim

#     class OrderDims(Transform):
#         """
#         Order dimensions of data
#         """

#         def apply(self, dataset: xr.Dataset) -> xr.Dataset:
#             _dim = dim
#             if missing == "skip":
#                 if isinstance(_dim, str) and _dim not in dataset.coords:
#                     _dim = None
#                 if isinstance(_dim, list):
#                     _dim = list(set(_dim).intersection(dataset.coords))

#             if as_dataarray:
#                 for var in dataset.data_vars:
#                     dataset[var] = dataset[var].expand_dims(_dim, axis=axis, **kwargs)
#                 return dataset
#             data = dataset.expand_dims(**kwargs)
#             return data

#         @property
#         def _info_(self) -> Any | dict:
#             return dict(dim=dim, axis=axis, as_dataarray=as_dataarray, missing=missing, **kwargs)

#     return ExpandDims()
