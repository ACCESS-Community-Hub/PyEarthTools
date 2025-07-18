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
Transform to apply to values of datasets
"""

from __future__ import annotations
from typing import Literal

import xarray as xr
import numpy as np

import pyearthtools.data
from pyearthtools.data.transforms.transform import Transform
from pyearthtools.utils.decorators import BackwardsCompatibility


class Fill(Transform):
    def __init__(
        self,
        coordinates: str | list[str],
        *coords,
        direction: Literal["forward", "backward", "both"] = "forward",
        limit: int | None = None,
    ):
        """
        Apply ffill or bfill on a dataset depending on given `direction`

        Args:
            coordinates (str | list[str]):
                Coordinates to run fill on
            direction (Literal["forward", "backward", "both"], optional):
                Direction to apply fill, either ffill or bfill. Defaults to 'forward'.
            limit (int | None, optional):
                limit to pass to fill. Defaults to None.
        """
        super().__init__()
        self.record_initialisation()

        coordinates = coordinates if isinstance(coordinates, (list, tuple)) else [coordinates]
        coordinates = [*coords, *coordinates]

        self._coordinates = coordinates
        self._direction = direction
        self._limit = limit

    def apply(self, dataset: xr.Dataset) -> xr.Dataset:
        encod = pyearthtools.data.transforms.attributes.set_encoding(reference=dataset)
        for coord in self._coordinates:
            if self._direction not in ["both", "forward", "backward"]:
                raise ValueError(f"Cannot parse {self._direction!r}, must be either 'forward', 'backward' or 'both'.")
            if self._direction in ["both", "forward"]:
                dataset = dataset.ffill(coord, limit=self._limit)
            if self._direction in ["both", "backward"]:
                dataset = dataset.bfill(coord, limit=self._limit)
        return encod(dataset)


class SetMissingToNaN(Transform):
    """
    Transform to replace specified missing values with NaN for given variables.

    Args:
        varname_val_map (dict[str, float]): A dictionary mapping variable names to their missing values.
    """

    def __init__(self, varname_val_map: dict[str, float]):
        super().__init__()
        self.record_initialisation()

        self.varname_val_map = varname_val_map

    def apply(self, dataset: xr.Dataset) -> xr.Dataset:
        for var_name, miss_val in self.varname_val_map.items():
            if var_name in dataset:
                dataset[var_name] = dataset[var_name].where(dataset[var_name] != miss_val, np.nan)
        return dataset


class AddFlaggedObs(Transform):
    """
    Transform to restore flagged observations removed by QC tests back into the dataset.

    Args:
        flagged_labels (list[str]): A list of variable names corresponding to flagged observations.
    """

    def __init__(self, flagged_labels: list[str]):
        super().__init__()
        self.record_initialisation()
        self.flagged_labels = flagged_labels

    def apply(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Apply the transformation to restore flagged observations.

        Args:
            dataset (xr.Dataset): The input dataset containing flagged_obs and QC'd variables.

        Returns:
            xr.Dataset: The dataset with flagged observations restored.
        """
        # Attach a coordinate to the flagged dimension with descriptive labels
        dataset = dataset.assign_coords(flagged=("flagged", self.flagged_labels))

        # Iterate through flagged variables and restore flagged data
        for var_name in dataset["flagged"].data:
            flagged_var = dataset["flagged_obs"].sel(flagged=var_name)
            qcd_var = dataset[var_name]

            # Fill NaNs in flagged variable with data from QC'd variable
            dataset[var_name].data = flagged_var.fillna(qcd_var).data

            #  # not all flagged values are available in flagged obs. Why?
            # # TODO: understand why and replace with observations if possible
            # dataset[var_name][dataset[var_name] == dataset[var_name].attrs['flagged_value']] = np.nan

            # Replace flagged placeholder values with NaN
            if "flagged_value" in dataset[var_name].attrs:
                # Compute the boolean condition to avoid Dask-related issues
                mask = dataset[var_name] == dataset[var_name].attrs["flagged_value"]
                dataset[var_name] = dataset[var_name].where(~mask, np.nan)

        return dataset


@BackwardsCompatibility(Fill)
def fill(*args, **kwargs): ...


def ffill(*a, **b):
    return Fill(*a, direction=b.pop("direction", "forward"), **b)


def bfill(*a, **b):
    return Fill(*a, direction=b.pop("direction", "backward"), **b)
