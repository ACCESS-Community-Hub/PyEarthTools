# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

from __future__ import annotations
from typing import Literal


import xarray as xr
from edit.data.transform.transform import Transform, TransformCollection


def rechunk(method: int | dict | Literal["auto", "encoding"]):
    if not isinstance(method, (int, dict)) and method not in ["auto", "encoding"] and method is not None:
        raise ValueError(f"method must be an int, dict, 'auto', 'encoding', or None. Instead found {method}.")

    class ReChunk(Transform):
        """Rechunk data"""

        def apply(self, dataset: xr.Dataset) -> xr.Dataset:
            for var in dataset:
                chunks = method
                if chunks == "encoding":
                    if "chunksizes" in dataset[var].encoding:
                        chunks = dataset[var].encoding["chunksizes"] or "auto"
                    else:
                        raise ValueError(f"Could not find 'chunksizes' in encoding of {var}")
                dataset[var].data = dataset[var].data.rechunk(chunks)
            return dataset

    return ReChunk()
