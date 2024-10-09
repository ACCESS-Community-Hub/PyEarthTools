# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations
from typing import Literal, Any


import xarray as xr
from edit.data.transforms.transform import Transform

from edit.utils.decorators import BackwardsCompatibility


class Rechunk(Transform):
    """Rechunk data"""

    def __init__(self, method: int | dict[str, Any] | Literal["auto", "encoding"]):
        """
        Rechunk data

        Args:
            method (int | dict[str, Any] | Literal['auto', 'encoding']):
                Rechunk either by encoding, auto or by variable config.
        """
        if not isinstance(method, (int, dict)) and method not in ["auto", "encoding"] and method is not None:
            raise ValueError(f"method must be an int, dict, 'auto', 'encoding', or None. Instead found {method}.")
        self._method = method

    def apply(self, dataset: xr.Dataset) -> xr.Dataset:
        for var in dataset:
            chunks = self._method
            if chunks == "encoding":
                if "chunksizes" in dataset[var].encoding:
                    chunks = dataset[var].encoding["chunksizes"] or "auto"
                else:
                    raise ValueError(f"Could not find 'chunksizes' in encoding of {var}")
            dataset[var].data = dataset[var].data.rechunk(chunks)
        return dataset


@BackwardsCompatibility(Rechunk)
def rechunk(*args, **kwargs): ...
