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

    def __init__(
        self,
        station: str,
        variables: list[str] | str | None = None,  # Ensure this is defined
        *,
        transforms: Transform | TransformCollection | None = None,  # Ensure this is keyword-only
    ):
        """
        Setup HadISD Indexer

        Args:
            station (str): Station ID to retrieve data for.
            transforms (optional): Base transforms to apply.
        """
        self.station = station
        self.variables = [variables] if isinstance(variables, str) else variables

        
        # Define the base transforms
        base_transform = TransformCollection()

        # Add a transform to drop unused variables (if variables are provided) REMOVE ONCE TESTED!!!!!!
        if variables:
            base_transform += pyearthtools.data.transforms.variables.Select(self.variables)

        # Add the variable selection transform, and any other transforms you want to apply
        # Future code to do this goes here, but the selcet class for variable doesn't exist yet, should look like this
        # if variables:
        #     base_transform += pyearthtools.data.transforms.variables.Select(
        #         {var: self.variable for var in ["variable"]}, ignore_missing=True
        #     )


        # Call the parent class's __init__ method
        super().__init__(
            transforms=base_transform + (transforms or TransformCollection()),
        )
       
        self.record_initialisation()



    def filesystem(self, *args, **kwargs) -> Path:
        """
        Map a query (station ID) to the corresponding file.

        Returns:
            Path: Path to the corresponding file.

        Raises:
            DataNotFoundError: If the file is not found.
        """
        HADISD_HOME = self.ROOT_DIRECTORIES["hadisd"]

        # Determine the parent folder based on the station ID
        station_id = self.station
        wmo_number = station_id[:6]  # Extract the first 6 digits of the station ID for determining the WMO number and parent folder

        # Define the station ranges and corresponding folders
        STATION_RANGES = [
            (0, 29999, "WMO_000000-029999"),
            (30000, 49999, "WMO_030000-049999"),
            (50000, 79999, "WMO_050000-079999"),
            (80000, 99999, "WMO_080000-099999"),
            (100000, 149999, "WMO_100000-149999"),
            (150000, 199999, "WMO_150000-199999"),
            (200000, 249999, "WMO_200000-249999"),
            (250000, 299999, "WMO_250000-299999"),
            (300000, 349999, "WMO_300000-349999"),
            (350000, 399999, "WMO_350000-399999"),
            (400000, 449999, "WMO_400000-449999"),
            (450000, 499999, "WMO_450000-499999"),
            (500000, 549999, "WMO_500000-549999"),
            (550000, 599999, "WMO_550000-599999"),
            (600000, 649999, "WMO_600000-649999"),
            (650000, 699999, "WMO_650000-699999"),
            (700000, 709999, "WMO_700000-709999"),
            (710000, 714999, "WMO_710000-714999"),
            (715000, 719999, "WMO_715000-719999"),
            (720000, 721999, "WMO_720000-721999"),
            (722000, 722999, "WMO_722000-722999"),
            (723000, 723999, "WMO_723000-723999"),
            (724000, 724999, "WMO_724000-724999"),
            (725000, 725999, "WMO_725000-725999"),
            (726000, 726999, "WMO_726000-726999"),
            (727000, 729999, "WMO_727000-729999"),
            (730000, 799999, "WMO_730000-799999"),
            (800000, 849999, "WMO_800000-849999"),
            (850000, 899999, "WMO_850000-899999"),
            (900000, 949999, "WMO_900000-949999"),
            (950000, 999999, "WMO_950000-999999"),
        ]

        # Find the parent folder dynamically
        parent_folder = None
        station_numeric = int(wmo_number)  # Convert the WMO number to an integer (I think WMO is just first 6 digitis of station ID?)
        for start, end, folder in STATION_RANGES:
            if start <= station_numeric <= end:
                parent_folder = folder
                break

        if parent_folder is None:
            raise ValueError(f"Station ID {station_id} does not fall within any defined range.")

        # Construct the expected filename
        date_range = "19310101-20240101"  # Hardcoded for now; adjust if dataset is updated
        version = "hadisd.3.4.0.2023f"
        filename = f"{version}_{date_range}_{station_id}.nc"

        # Construct the full path
        file_path = Path(HADISD_HOME) / parent_folder / filename

        # Check if the file exists
        if not file_path.exists():
            raise DataNotFoundError(
                f"File not found for station: {station_id}, path: {file_path}"
            )

        # Return the constructed file path
        return file_path

    @property
    def _import(self):
        """module to import for to load this step in an Pipeline"""
        return "pyearthtools.tutorial"



# Notes to Joel
# - Does PET have the ability to check a NetCDF file for the variables it contains?
# - If not, we should add that to PET so that a user can be given suggestions for what variables to select, should they give an incorrect variable name
# - # Convert querytime to Petdt for consistency. Useful for other classes that may not have the same conversion
       # querytime = Petdt(querytime)