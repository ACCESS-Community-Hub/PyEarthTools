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
Binning of datasets by dimension and predefined configurations.
"""

from __future__ import annotations

import xarray as xr
from typing import Literal

from pyearthtools.data import TimeDelta


BINNING_SETUP = {  # Base Binning setup
    "seasonal": [
        *(TimeDelta(i, "days") for i in range(0, 7)),
        TimeDelta(7, "days"),
        TimeDelta(14, "days"),
    ],
    "daily": [TimeDelta(0, "days"), TimeDelta(1, "days")],
    "weekly": [TimeDelta(0, "days"), TimeDelta(7, "days")],
}
DELTA = {  # Delta to expand bins by
    "seasonal": TimeDelta(7, "days"),
    "daily": TimeDelta(1, "days"),
    "weekly": TimeDelta(7, "days"),
}


def binning(
    data: xr.Dataset | xr.DataArray,
    setup: Literal[tuple(BINNING_SETUP.keys())],
    *,
    dimension: str = "time",
    expand: bool = True,
    offset: int | str | TimeDelta | None = None,
) -> "xr.DatasetGroupBy | xr.DataArrayGroupBy":
    """
    Bin `data` based on a binning setup.

    If `expand` is `True` use `DELTA` to create new bins until all included.

    ## Implemented:
    | name | Description |
    | ---- | ----------- |
    | seasonal | Daily up till first week, than weekly |
    | daily | Daily grouping |
    | weekly | Weekly grouping |

    Args:
        data (xr.Dataset | xr.DataArray):
            Data to bin
        setup (str):
            Binning config to use.
        dimension (str, optional):
            Dimension to bin across. Defaults to 'time'.
        expand (bool, optional):
            Whether to expand bins to encompass all the data. Defaults to True.
        offset (int | TimeDelta | None, optional):
            Offset to add to starting time. Will be the minimum value
            upon `time` axis. Defaults to None.

    Raises:
        ValueError:
            If `setup` not available, or not in `DELTA` while `expand` is True.
        AttributeError:
            If `dimension` not in `data`.

    Returns:
        (xr.DatasetGroupBy | xr.DataArrayGroupBy):
            Data binned according to config.
    """

    if setup not in BINNING_SETUP:
        raise ValueError(f"Cannot parse setup: {setup}. Valid are: {list(BINNING_SETUP.keys())}")

    if dimension not in data.dims:
        raise AttributeError(f"Cannot groupby dimension {dimension!r}, when data contains {data.dims}. Set `dimension`")

    min_value = data[dimension].min()
    if offset is not None:
        min_value += TimeDelta(offset) if isinstance(offset, str) else offset
    delta = (data[dimension].max() - min_value).values

    bins = BINNING_SETUP[setup]

    if expand:
        if setup not in DELTA and bins[-1] < delta:
            raise ValueError(f"As {setup!r} does not have a defined `DELTA` cannot expand bins.")

        while bins[-1] < delta:
            bins.append(bins[-1] + DELTA[setup])

    bins = [min_value.values + x for x in bins]

    return data.groupby_bins(dimension, bins)
