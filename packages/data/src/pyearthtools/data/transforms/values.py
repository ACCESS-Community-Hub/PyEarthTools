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


@BackwardsCompatibility(Fill)
def fill(*args, **kwargs): ...


def ffill(*a, **b):
    return Fill(*a, direction=b.pop("direction", "forward"), **b)


def bfill(*a, **b):
    return Fill(*a, direction=b.pop("direction", "backward"), **b)
