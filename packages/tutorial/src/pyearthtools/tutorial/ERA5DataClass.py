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
ECWMF ReAnalysis v5, Low-Resolution / WeatherBench Example

The purpose of this module is to hold the index class which can be
registered into the pyearthtools package namespace for easy access.

The code here is the interface between the pyearthtools API and accessing
files on the filesystem.

An indexer takes some
"""

from __future__ import annotations

import functools
from pathlib import Path
import xarray as xr
import os.path

import pyearthtools.data
from pyearthtools.data import Petdt
from pyearthtools.data.archive import register_archive
from pyearthtools.data.exceptions import DataNotFoundError
from pyearthtools.data.indexes import ArchiveIndex, decorators
from pyearthtools.data.transforms import Transform, TransformCollection


# This tells pyearthtools what the actual resolution or time-step of the data is inside the files
ERA_RESOLUTION = (1, "hour")
ERADEMO_RESOLUTION = (6, "hour")

# This dictionary tells pyearthtools what variable renames to apply during load
ERA5_RENAME = {"t2m": "2t", "u10": "10u", "v10": "10v", "siconc": "ci"}

ERA5DEMO_RENAME = {
    "t2m": "2t",
    "10m_u_component_of_wind": "10u",
    "10m_v_component_of_wind": "10v",
    "mean_sea_level_pressure": "mslp",
}

V_TO_PATH = {
    "10m_u_component_of_wind": "10m_u_component_of_wind",
    "10m_v_component_of_wind": "10m_v_component_of_wind",
    "2m_temperature": "2m_temperature",
    # "constants": "constants",  # FIXME not working
    "geopotential": "geopotential",
    # "geopotential_500": "geopotential_500",  # FIXME not working
    "potential_vorticity": "potential_vorticity",
    "rh": "relative_humidity",
    "specific_humidity": "specific_humidity",
    "temperature": "temperature",
    # "temperature_850": "temperature_850",  # FIXME not working
    "toa_incident_solar_radiation": "toa_incident_solar_radiation",
    "total_cloud_cover": "total_cloud_cover",
    "total_precipitation": "total_precipitation",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "vorticity": "vorticity",
}


@functools.lru_cache()
def cached_iterdir(path: Path) -> list[Path]:
    """Run iterdir but cached"""
    return list(path.iterdir())


@functools.lru_cache()
def cached_exists(path: Path) -> bool:
    """Run exits but cached"""
    return path.exists()


@register_archive("era5lowres", sample_kwargs=dict(variable="2t"))
class ERA5LowResIndex(ArchiveIndex):
    """ECWMF ReAnalysis v5"""

    @property
    def _desc_(self):
        return {
            "singleline": "ECWMF ReAnalysis v5",
            "range": "1970-current",
            "Documentation": "https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation",
        }

    @decorators.alias_arguments(
        level_value=["pressure"],
        variables=["variable"],
        product=["resolution"],
    )
    @decorators.variable_modifications(variable_keyword="variables", remove_variables=False)
    @decorators.deprecated_arguments(
        level="`level` is deprecated in the ERA5 index. Simply provide the variables, `level` will be autofound."
    )
    def __init__(
        self,
        variables: list[str] | str,
        *,
        level_value: int | float | list[int | float] | tuple[list | int, ...] | None = None,
        transforms: Transform | TransformCollection | None = None,
    ):
        """
        Setup ERA5 Low-Res Indexer

        Args:
            variables (list[str] | str):
                Data variables to retrieve
            resolution (Literal[ERA_RES], optional):
                Resolution of data, must be one of 'monthly-averaged','monthly-averaged-by-hour', 'reanalysis'.
                Defaults to 'reanalysis'.
            level_value: (int, optional):
                Level value to select if data contains levels. Defaults to None.
            transforms (Transform | TransformCollection, optional):
                Base Transforms to apply.
                Defaults to TransformCollection().
        """

        variables = [variables] if isinstance(variables, str) else variables

        self.resolution = ERA_RESOLUTION

        self.variables = variables
        base_transform = TransformCollection()

        # base_transform += pyearthtools.data.transforms.attributes.Rename(ERA5_RENAME)
        # base_transform += pyearthtools.data.transforms.variables.Trim(variables)

        self.level_value = level_value

        if level_value:
            base_transform += pyearthtools.data.transforms.coordinates.Select(
                {coord: level_value for coord in ["level"]}, ignore_missing=True
            )

        super().__init__(
            transforms=base_transform + (transforms or TransformCollection()),
            data_interval=ERA_RESOLUTION,
        )
        self.record_initialisation()

    def filesystem(
        self,
        querytime: str | Petdt,
    ) -> Path | dict[str, str | Path]:
        ERA5_HOME = self.ROOT_DIRECTORIES["era5lowres"]

        """
        This tells pyearthtools how to go from a request for a date/time to a path containing the files
        which will match that request.
        """

        paths = {}
        querytime = Petdt(querytime)

        for variable in self.variables:

            var_path = Path(ERA5_HOME) / V_TO_PATH[variable]

            files_in_dir = cached_iterdir(var_path)
            start_of_month_string = querytime.strftime("%Y")

            relevant_path = None
            for filename in files_in_dir:
                if start_of_month_string in str(filename):
                    relevant_path = filename
                    break

            if relevant_path is not None:
                relevant_path = var_path / relevant_path

                if cached_exists(relevant_path):
                    paths[variable] = relevant_path
                    continue

            raise DataNotFoundError(
                f"Unable to find data for: basetime: {querytime}, variables: {variable} at {var_path}"
            )
        return paths

    @property
    def _import(self):
        """module to import for to load this step in an Pipeline"""
        return "pyearthtools.tutorial"


@register_archive("era5_demo_subset", sample_kwargs=dict(variable="2t"))
class ERA5LowResDemoIndex(ArchiveIndex):
    """ECWMF ReAnalysis v5"""

    @property
    def _desc_(self):
        return {
            "singleline": "ECWMF ReAnalysis v5",
            "range": "1970-current",
            "Documentation": "https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation",
        }

    @decorators.alias_arguments(
        level_value=["pressure"],
        variables=["variables"],
    )
    @decorators.variable_modifications(variable_keyword="variables", remove_variables=False)
    @decorators.deprecated_arguments(
        level="`level` is deprecated in the ERA5 index. Simply provide the variables, `level` will be autofound."
    )
    def __init__(
        self,
        variables: list[str] | str,
        *,
        level_value: int | float | list[int | float] | tuple[list | int, ...] | None = None,
        transforms: Transform | TransformCollection | None = None,
        filename_override=None,
        product=None,
    ):
        """
        Setup ERA5 Low-Res Indexer

        Args:
            variables (list[str] | str):
                Data variables to retrieve
            resolution (Literal[ERA_RES], optional):
                Resolution of data, must be one of 'monthly-averaged','monthly-averaged-by-hour', 'reanalysis'.
                Defaults to 'reanalysis'.
            level_value: (int, optional):
                Level value to select if data contains levels. Defaults to None.
            transforms (Transform | TransformCollection, optional):
                Base Transforms to apply.
                Defaults to TransformCollection().
        """

        variables = [variables] if isinstance(variables, str) else variables

        self.resolution = ERADEMO_RESOLUTION
        self.dataset = None

        self.filename_override = filename_override

        self.variables = variables
        base_transforms = TransformCollection()

        # base_transforms += pyearthtools.data.transforms.attributes.Rename(ERA5DEMO_RENAME)
        base_transforms += pyearthtools.data.transforms.variables.Trim(variables)

        self.level_value = level_value

        if level_value:
            base_transforms += pyearthtools.data.transforms.coordinates.Select(
                {coord: level_value for coord in ["level"]}, ignore_missing=True
            )

        # Add in any user-supplied transforms
        if transforms is not None:
            base_transforms += transforms

        super().__init__(
            base_transforms,
            data_interval=ERADEMO_RESOLUTION,
        )
        self.record_initialisation()

    def filesystem(
        self,
        querytime: str | Petdt,
    ) -> Path | dict[str, str | Path]:
        ERA5_HOME = self.ROOT_DIRECTORIES["era5lowresdemo"]

        """
        This tells pyearthtools how to go from a request for a date/time to a path containing the files
        which will match that request.
        """

        if self.filename_override is not None:
            path = self.filename_override

        else:
            path = Path(ERA5_HOME) / "era5_lowres.nc"  # Everything fits into a single 2 GIG file

        if not os.path.exists(path):
            raise ValueError(f"Could not find a matching input file at {path}")

        return [path]

    def load(self, *args, **kwargs):
        """
        This particular example has all its data in a single file, so repeatedly
        loading files is avoided through caching the loaded dataset. This isn't
        a great general pattern, but works well for the tutorial.
        """

        if self.dataset:
            return self.dataset

        ds = xr.open_dataset(args[0][0], engine="h5netcdf")
        self.dataset = ds

        return self.dataset

    @property
    def _import(self):
        """module to import for to load this step in an Pipeline"""
        return "pyearthtools.tutorial"

    @classmethod
    def sample(cls):
        return ERA5LowResDemoIndex("2m_temperature")
