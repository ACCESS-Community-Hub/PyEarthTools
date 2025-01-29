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
Provide common operations to be applied to [DataIndexes][pyearthtools.data.DataIndex]
"""

from __future__ import annotations

import datetime
import builtins
import os
from pathlib import Path
from typing import Callable, Union

import numpy as np
from tqdm.auto import tqdm, trange
import xarray as xr

import pyearthtools.data

from pyearthtools.data.time import pyearthtoolsDatetime, TimeDelta
from pyearthtools.data.exceptions import (
    DataNotFoundError,
    InvalidIndexError,
    run_and_catch_exception,
)
from pyearthtools.data.transforms import Transform, TransformCollection


def split_ds(dataset: xr.Dataset, divisions: int = 1, dim: str = "time") -> list[xr.Dataset]:
    """
    Split an xarray Dataset into a set number of datasets

    Args:
        dataset (xr.Dataset): Dataset to split
        divisions (int, optional): Number of divisions to make. Defaults to 1.
        dim (str, optional): Which dim to split on. Defaults to "time".

    Returns:
        list[xr.Dataset]: List of Datasets
    """

    return list(split_ds_gen(dataset, divisions, dim))


def split_ds_gen(dataset: xr.Dataset, divisions: int = 1, dim: str = "time") -> list[xr.Dataset]:
    """
    Generator version of split_ds

    Args:
        dataset (xr.Dataset): Dataset to split
        divisions (int, optional): Number of divisions to make. Defaults to 1.
        dim (str, optional): Which dim to split on. Defaults to "time".

    Yields:
        list[xr.Dataset]: List of Datasets
    """

    samples = len(dataset[dim])
    divisions = min(divisions, samples)
    per_divison = samples // divisions

    for i in range(divisions):
        yield dataset.isel(**{dim: slice(per_divison * i, per_divison * (i + 1))})


def aggregation(
    DataFunction: "TimeIndex",
    start: str | datetime.datetime | pyearthtoolsDatetime,
    end: str | datetime.datetime | pyearthtoolsDatetime,
    interval: tuple[float, str],
    *,
    aggregation: str = "mean",
    aggregation_dim: str = "time",
    save_location: str | Path | None = None,
    skip_invalid: bool = True,
    num_divisions: int = 1,
    transforms: Transform | TransformCollection = TransformCollection(),
    verbose: bool = False,
    **kwargs,
) -> xr.Dataset:
    """
    Get aggregation of [TimeIndex][pyearthtools.data.TimeIndex] over given dimension

    !!! Warning:
        Any `num_divisions` not a factor of the number of data steps
        will result in some data being missed.

    Args:
        DataFunction (TimeIndex):
            TimeIndex to retrieve Data
        start (str | datetime.datetime | pyearthtoolsDatetime):
            Start Date
        end (str | datetime.datetime | pyearthtoolsDatetime):
            End Date
        interval (tuple[float, str]):
            Interval between samples. Use pandas.to_timedelta notation, (10, 'minute')
        aggregation (str, optional):
            Aggregation Function to apply. Defaults to "mean".
        aggregation_dim (str, optional):
            Dimension to aggregate over apply. Defaults to "time".
        save_location (str | Path | None, optional):
            Location to automatically save the result. Defaults to None.
        skip_invalid (bool, optional):
            Whether to skip invalid data. Defaults to True.
        num_divisions (int, optional):
            Number of times to divide series to alleviate memory issues. Defaults to 1.
        transforms (Transform | TransformCollection, optional):
            Extra Transforms to be applied. Defaults to TransformCollection().
        verbose (bool, optional):
            Whether to log progress messages. Defaults to False.

    Returns:
        xr.Dataset: Dataset with aggregation applied
    """
    print = lambda *args, **kwargs: builtins.print(*args, **kwargs) if verbose else None

    # print("Finding Series ...")
    aggregation_func = pyearthtools.data.transforms.aggregation.over(aggregation, dimension=aggregation_dim)

    start = pyearthtoolsDatetime(start)
    end = pyearthtoolsDatetime(end)
    interval = TimeDelta(interval)
    steps = (end - start) // interval

    if num_divisions > 1:
        components = []
        print(f"Finding Series in {num_divisions} divisions ...")

        for i in trange(
            num_divisions,
            disable=not verbose,
            desc=f"Calculating {aggregation}",
        ):
            start_time = start + (interval * ((steps // num_divisions) * i))
            end_time = start + (interval * ((steps // num_divisions) * (i + 1)))
            try:
                dataset = DataFunction.series(
                    start_time,
                    end_time,
                    skip_invalid=skip_invalid,
                    interval=interval,
                    transforms=transforms,
                    verbose=False,
                    **kwargs,
                )
                if dataset is None:
                    continue

                aggregated_data = aggregation_func(dataset)
                if save_location:
                    component_location = (
                        Path(save_location).with_suffix("") / f"components/temp_file_{start_time}-{end_time}.nc"
                    )
                    if not component_location.parent.exists():
                        component_location.parent.mkdir(exist_ok=True, parents=True)
                    aggregated_data.to_netcdf(component_location)
                    aggregated_data = xr.open_dataset(component_location)

                components.append(aggregated_data)
            except DataNotFoundError as e:
                if not skip_invalid:
                    raise e
                print(str(e))

        print("Concatenating Data")
        dataset = aggregation_func(xr.concat(components, dim="time").sortby("time"))

    else:
        print("Finding Series ...")

        dataset = DataFunction.series(
            start,
            end,
            skip_invalid=skip_invalid,
            interval=interval,
            transforms=transforms,
            verbose=verbose,
            **kwargs,
        )

        dataset = aggregation_func(dataset)

    # print(f"Found Series of size: ", len(dataset.time))

    if save_location is None:
        save_file = None
    else:
        print("Saving Data")
        dataset = dataset.compute()

        save_file = Path(save_location)
        if not save_file.parent.exists():
            save_file.parent.mkdir(parents=True, exist_ok=True)

    dataset.attrs.update({"Aggregation": f"Aggregation {aggregation} from {start} to {end} at {interval} intervals"})
    for var in dataset.data_vars:
        dataset[var].attrs.update(
            {"Aggregation": f"Aggregation {aggregation} from {start} to {end} at {interval} intervals"}
        )

    if save_file is not None:
        dataset.to_netcdf(save_file, mode="w")
        for file in (Path(save_location).with_suffix("") / f"components/").glob("temp_file_*.nc"):
            os.remove(file)
    return dataset


MIN_VALUE = 1e10
MAX_VALUE = 1e-10


def find_range(
    DataFunction: "TimeIndex",
    start: str | pyearthtoolsDatetime,
    end: str | pyearthtoolsDatetime,
    interval: tuple[float, str] | TimeDelta,
    *,
    skip_invalid: bool = True,
    num_divisions: int = 1,
    transforms: Transform | TransformCollection = TransformCollection(),
    **kwargs,
) -> dict:
    """
    Find Minimum and Maximum of a [TimeIndex][pyearthtools.data.TimeIndex] in the given time range

    Args:
        DataFunction (TimeIndex):
            TimeIndex to retrieve Data
        start (str | pyearthtoolsDatetime):
            Start Date
        end (str | pyearthtoolsDatetime):
            End Date
        interval (tuple[float, str]):
            Interval between samples. Use pandas.to_timedelta notation, (10, 'minute')
        skip_invalid (bool, optional):
            Whether to skip invalid data. Defaults to True.
        num_divisions (int, optional):
            Number of times to divide series to alleviate memory issues. Defaults to 1.
        transforms (Transform | TransformCollection, optional):
            Extra Transforms to be applied. Defaults to TransformCollection().

    Returns:
        dict: Dictionary with max and min populated
    """
    dataset = DataFunction.series(
        start,
        end,
        skip_invalid=skip_invalid,
        interval=interval,
        transforms=transforms,
        **kwargs,
    )

    split_dataset = split_ds(dataset, num_divisions)

    max_values = (
        xr.concat([ds.sortby("time").max(skipna=True) for ds in split_dataset], dim="time")
        .sortby("time")
        .max(dim="time", skipna=True)
    )
    min_values = (
        xr.concat([ds.sortby("time").min(skipna=True) for ds in split_dataset], dim="time")
        .sortby("time")
        .min(dim="time", skipna=True)
    )

    return {var: {"max": max_values[var].data, "min": min_values[var].data} for var in max_values}
