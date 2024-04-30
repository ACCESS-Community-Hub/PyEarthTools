# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

"""
Wrap [np.percentile][numpy.percentile] to work on xarray Datasets/DataArrays
"""

from __future__ import annotations

from typing import Iterable, Union

import numpy as np
import xarray as xr


def _find_percentile(data: xr.DataArray | xr.Dataset, percentiles: float | list[float]):
    if isinstance(data, xr.Dataset):
        return tuple(map(_find_percentile, data))  # type: ignore
    return np.nanpercentile(data, percentiles)


def percentile(dataset: xr.DataArray | xr.Dataset, percentiles: float | list[float]) -> xr.Dataset:
    """
    Find Percentiles of given data

    Args:
        dataset (xr.DataArray | xr.Dataset): Dataset to find percentiles of
        percentiles (float | list[float]): Percentiles to find either float or list[float]

    Returns:
        (xr.Dataset): Dataset with percentiles

    Examples:
        >>> percentile(dataset, [1, 99])
        # Dataset containing 1st and 99th percentiles

    """
    if not isinstance(percentiles, Iterable):
        percentiles = [percentiles]

    if isinstance(dataset, xr.DataArray):
        dataset = dataset.to_dataset()

    new_data = {}
    coords = {"Percentile": percentiles}

    for data_var in dataset.data_vars:
        new_data[data_var] = (
            coords,
            _find_percentile(dataset[data_var], percentiles),
            dataset[data_var].attrs,
        )

    return xr.Dataset(data_vars=new_data, coords=coords)
