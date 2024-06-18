# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
Add variables
"""
from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr
from edit.data import Transform


class TimeOfYear(Transform):
    """
    Add time of year to dataset

    """

    def __init__(self, method: str):
        """
        Add time of year as variable to a dataset

        Use [DropDataset][edit.pipeline.operations.select.DropDataset] to remove it
        if an earlier step in the pipeline is sensitive to variable names.

        Args:
            method (str):
                Method to use, either "dayofyear" or "monthofyear"
                Both modelled as a sinusodal function

        Returns:
            (Transform):
                Transform to add time of year variable
        """
        if method not in ["dayofyear", "monthofyear"]:
            raise ValueError(f"Invalid method passed, cannot be {method!r}. Must be in ['dayofyear', 'monthofyear']")
        self.method = method

    def apply(self, ds: xr.Dataset):
        dims = ds.dims

        if self.method == "dayofyear":
            value = (np.cos(ds.time.dt.dayofyear * np.pi / (366 / 2)) + 1) / 2
        if self.method == "monthofyear":
            value = (np.cos(ds.time.dt.date.month * np.pi / 6) + 1) / 2

        new_dims = {}

        for key in (key for key in dims.keys() if key not in ["time"]):
            new_dims[key] = np.atleast_1d(ds[key].values)

        axis = [list(dims).index(key) for key in new_dims.keys()]
        ds[self.method] = value.expand_dims(new_dims, axis=axis)

        # value = value * np.ones([len(ds[dim]) for dim in list(ds.dims)])
        # ds[method] = (ds.dims, value)

        dims = ds[list(ds.data_vars)[0]].dims

        ds = ds.transpose(*list(dims))
        return ds

    @property
    def _info_(self) -> Any | dict:
        return dict(method=self.method)
