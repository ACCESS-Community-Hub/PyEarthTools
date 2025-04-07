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


from __future__ import annotations

import numpy as np
import xarray as xr


def fake_dataset(
    variables: str | list[str],
    time_labels: list[str],
    size: tuple[int] = (256, 256),
    fill_value: str | int = 1,
):
    """
    Create a Fake Dataset for use in Testing
    Full customisation of variable names, time_labels and size

    Parameters
    ----------
    variables
        Variable names to create
    time_labels
        Values for the time dimension
    size, optional
        Size of lat/lon, by default (256,256)
    fill_value, optional
        Value to fill array with, use 'random' for random values, by default 1

    Returns
    -------
        xr.Dataset of specified shape and values
    """

    if not isinstance(variables, (list, tuple)):
        variables = [variables]
    if not isinstance(time_labels, (list, tuple)):
        time_labels = [time_labels]

    if fill_value == "random":
        fake_data = np.random.random((len(variables), len(time_labels), *size))
    else:
        fake_data = np.full((len(variables), len(time_labels), *size), fill_value=fill_value)

    fake_ds = xr.Dataset(
        data_vars={var: (["time", "lat", "lon"], fake_data[i]) for i, var in enumerate(variables)},
        coords=dict(lon=range(0, size[0]), lat=range(0, size[1]), time=time_labels),
        attrs=dict(WARNING="Fake Data for Testing"),
    )
    return fake_ds
