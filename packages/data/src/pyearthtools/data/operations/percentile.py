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
