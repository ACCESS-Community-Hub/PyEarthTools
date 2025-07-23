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
ECWMF ReAnalysis v5
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal


import pyearthtools.data

from pyearthtools.data import Petdt
from pyearthtools.data.exceptions import DataNotFoundError
from pyearthtools.data.indexes import ArchiveIndex, decorators
from pyearthtools.data.transforms import Transform, TransformCollection
from pyearthtools.data.archive import register_archive

from site_archive_met_office.utilities import (
    cached_exists,
    cached_iterdir,
)  # Could these be moved into a generic module?
from site_archive_met_office.ancilliary.ERA5 import (
    ERA5_SINGLE_VARIABLES,
    ERA5_PRESSURE_VARIABLES,
)  # NOT NEEDED NOW?   # Could these be moved into a generic module?

ERA_PROD = ["monthly-averaged", "monthly-averaged-by-hour", "reanalysis"]
ERA_RESOLUTION = (1, "hour")

# ERA5_RENAME = {"t2m": "2t", "u10": "10u", "v10": "10v", "siconc": "ci"}                                 # Why is this needed?


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


@register_archive("ERA5lowres", sample_kwargs=dict(variable="2t"))
class ERA5lowres(ArchiveIndex):
    """ECWMF ReAnalysis v5 Low-Resolution / WeatherBench"""

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
        # base_transform += pyearthtools.data.transforms.variables.variable_trim(variables)

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
        ERA5lowres_HOME = self.ROOT_DIRECTORIES["ERA5lowres"]

        paths = {}
        querytime = Petdt(querytime)

        for variable in self.variables:

            # This line tells pyearthtools how to go from a request for a date/time to a path containing the files
            var_path = Path(ERA5lowres_HOME) / V_TO_PATH[variable]

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

    # Do we need this?
    @property
    def _import(self):
        """module to import when this class is used"""
        return "pyearthtools.site_archive_met_office.ERA5lowres"
