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
from typing import Any, Literal

import pyearthtools.data
from pyearthtools.data import Petdt
from pyearthtools.data.archive import register_archive
from pyearthtools.data.exceptions import DataNotFoundError
from pyearthtools.data.indexes import ArchiveIndex, decorators
from pyearthtools.data.transforms import Transform, TransformCollection

from pyearthtools.tutorial.ancilliary.ERA5lowres import (
    ERA5_PRESSURE_VARIABLES,
    ERA5_SINGLE_VARIABLES,
)

# This tells pyearthtools what the actual resolution or time-step of the data is inside the files
ERA_RESOLUTION = (1, "hour")

# This dictionary tells pyearthtools what variable renames to apply during load
ERA5_RENAME = {"t2m": "2t", "u10": "10u", "v10": "10v", "siconc": "ci"}

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


@register_archive("hadisd", sample_kwargs=dict(station="010010-99999"))
class HadISDIndex(ArchiveIndex):
    """HadISD Dataset Index"""

    @property
    def _desc_(self):
        return {
            "singleline": "HadISD Dataset",
            "range": "1931-2024",
            "Documentation": "https://www.metoffice.gov.uk/hadobs/hadisd/",
        }

    def __init__(self, station: str, *, transforms=None):
        """
        Setup HadISD Indexer

        Args:
            station (str): Station ID to retrieve data for.
            transforms (optional): Base transforms to apply.
        """
        self.station = station
        super().__init__(transforms=transforms)

    def filesystem(self, querytime: str | Petdt) -> Path:
        """
        Map a query (station ID and date) to the corresponding file.

        Args:
            querytime (str | Petdt): Query date.

        Returns:
            Path: Path to the corresponding file.

        Raises:
            DataNotFoundError: If the file is not found.
        """
        HADISD_HOME = self.ROOT_DIRECTORIES["hadisd"]

        # Convert querytime to Petdt for consistency
        querytime = Petdt(querytime)

        # Construct the expected filename pattern
        station_id = self.station
        date_range = "19310101-20240101"  # Hardcoded for now; adjust if needed
        version = "hadisd.3.4.0.2023f"

        filename = f"{version}_{date_range}_{station_id}.nc"
        file_path = Path(HADISD_HOME) / filename

        # Check if the file exists
        if not file_path.exists():
            raise DataNotFoundError(
                f"File not found for station: {station_id}, date: {querytime}, path: {file_path}"
            )

        return file_path

    @property
    def _import(self):
        """module to import for to load this step in an Pipeline"""
        return "pyearthtools.tutorial"



# Notes to Joel
# - Does PET have the ability to check a NetCDF file for the variables it contains?
# - If not, we should add that to PET so that a user can be given suggestions for what variables to select, should they give an incorrect variable name
