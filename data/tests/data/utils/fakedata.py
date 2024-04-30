# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

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
