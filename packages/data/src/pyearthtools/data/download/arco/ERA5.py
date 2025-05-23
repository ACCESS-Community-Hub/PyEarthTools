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

import xarray as xr

import pyearthtools.data
from pyearthtools.data.time import Petdt

from pyearthtools.data.indexes import AdvancedTimeDataIndex, decorators
from pyearthtools.data.transforms.transform import Transform, TransformCollection

from pyearthtools.data.download.arco._ERA5 import (
    ERA5_VARIABLES,
    ERA5_LEVELS,
    ERA_NAME_CHANGE,
)


def get_from_shortname(variables: list[str]) -> list[str]:
    """Convert from variable short name"""

    def invert_dict(dictionary: dict[str, str]) -> dict[str, str]:
        return {val: key for key, val in dictionary.items()}

    short_name = invert_dict(ERA_NAME_CHANGE)

    return [short_name[var] if var in short_name else var for var in variables]


def open_arco(variables, level=None, chunks="auto", **kwargs):
    """Open Analysis-Ready Cloud Optimized ERA5 archive from Google Cloud Platform"""
    ds = xr.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        chunks=chunks,
        storage_options=dict(token="anon"),
        **kwargs,
    )
    ds = ds[variables]

    if level is not None:
        ds = pyearthtools.data.transform.coordinates.Select(
            level=level, ignore_missing=True)(ds)

    return ds


class ARCOERA5(AdvancedTimeDataIndex):
    """
    Analysis-Ready, Cloud Optimized ERA5

    https://github.com/google-research/arco-era5

    Carver, Robert W, and Merose, Alex. (2023):
    ARCO-ERA5: An Analysis-Ready Cloud-Optimized Reanalysis Dataset.
    22nd Conf. on AI for Env. Science, Denver, CO, Amer. Meteo. Soc, 4A.1,
    https://ams.confex.com/ams/103ANNUAL/meetingapp.cgi/Paper/415842
    """

    _desc_ = {
        "singleline": "Analysis-Ready, Cloud Optimized ERA5",
        "link": "https://github.com/google-research/arco-era5",
    }

    @decorators.alias_arguments(
        variables=["variable"],
        level=["levels", "level_value"],
    )
    @decorators.check_arguments(
        variables=ERA5_VARIABLES,
        level=ERA5_LEVELS,
    )
    @decorators.variable_modifications("variables")
    def __init__(
        self,
        variables: str | list[str],
        level: int | list[int] | None = None,
        transforms: Transform | TransformCollection | None = None,
        **kwargs,
    ):
        """
        Analysis-Ready, Cloud Optimized ERA5 integrated within `pyearthtools`.

        Allows for access to a cloud ERA5 archive.

        Args:
            variables (str | list[str]):
                Variables to retrieve, can be either short_name or long_name
            level (int | list[int] | None, optional):
                Pressure levels to select. Defaults to None.
            transforms (Transform | TransformCollection | None, optional):
                Transforms to apply to dataset. Defaults to None.
        """
        super().__init__(transforms or TransformCollection(), data_interval="1 hour")
        self.record_initialisation()

        variables = get_from_shortname([variables] if not isinstance(variables, list) else variables)
        self.variables = variables
        self.level = level

        self._kwargs = kwargs
        self._ds = open_arco(variables, level, **kwargs)

    @property
    def dataset(self) -> xr.Dataset:
        """Get full dataset for this obj"""
        return self._ds

    def get(self, time: str):
        """Get timestep from dataset"""
        return self._ds.sel(time=Petdt(time).datetime64())

    @classmethod
    def sample(cls):
        return ARCOERA5("2m_temperature")
