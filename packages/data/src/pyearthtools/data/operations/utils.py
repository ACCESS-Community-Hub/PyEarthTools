from __future__ import annotations
from typing import Hashable

import xarray as xr

STANDARD_TIME_COORD_NAMES = ["time", "basetime"]


def identify_time_dimension(data: xr.DataArray | xr.Dataset) -> str:
    """Attempt to identify time dimension in dataset.

    If cannot be identified, return 'time'
    """
    coords = list(str(x) for x in data.coords)

    for coord in coords:
        if coord in STANDARD_TIME_COORD_NAMES:
            return coord

    for coord in coords:
        if "time" in coord:
            return coord

    for coord in coords:
        dtype = data[coord].dtype
        if "time" in dtype.__class__.__name__:
            return coord

    return "time"
