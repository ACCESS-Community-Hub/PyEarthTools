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
import xarray as xr
from pathlib import Path
from typing import Any, Literal

import pyearthtools.data
from pyearthtools.data import Petdt
from pyearthtools.data.archive import register_archive
from pyearthtools.data.exceptions import DataNotFoundError
from pyearthtools.data.indexes import ArchiveIndex, decorators
from pyearthtools.data.transforms import Transform, TransformCollection
from pyearthtools.data.transforms.variables import Drop, Select
from pyearthtools.data.transforms.values import SetMissingToNaN


# This dictionary tells pyearthtools which variables have missing values and what those values are.
varname_val_map = {
        "total_cloud_cover": -999., 
        "low_cloud_cover": -999., 
        "mid_cloud_cover": -999.,
        "high_cloud_cover": -999.
    }
# TODO:Check that these values actually represent missing values in the dataset


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
        station: str | list[str] | None = None,  # Allow single station, multiple stations, or None
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
        self.station = [station] if isinstance(station, str) else station
        self.variables = [variables] if isinstance(variables, str) else variables

        # Define the base transforms
        base_transform = TransformCollection()
        base_transform += Drop("reporting_stats" )

        # Add a transform to select variables (if variables are provided)
        if variables:
            base_transform += Select(self.variables)

        base_transform += SetMissingToNaN(varname_val_map)

        # Call the parent class's __init__ method
        super().__init__(
            transforms=base_transform + (transforms or TransformCollection()),
        )
       
        self.record_initialisation()


    def filesystem(self, station_ids: str | list[str], *args, **kwargs) -> dict[str, Path]:
        """
        Map a stations ID or list of station IDs to their corresponding file paths.

        Args:
            station (str | list[str]): Station ID or list of station IDs.

        Returns:
            dict[str, Path]: A dictionary mapping station IDs to their corresponding file paths.

        Raises:
            DataNotFoundError: If a file is not found for any station ID.
        """
        HADISD_HOME = self.ROOT_DIRECTORIES["hadisd"]

        # If string is given, convert to list for simpler/consistent handling
        if isinstance(station_ids, str):
            station_ids = [station_ids]

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

        # Map station IDs to their file paths
        paths = {}
        for station_id in station_ids:
            wmo_number = station_id[:6]  # Extract the first 6 digits of the station ID
            station_numeric = int(wmo_number)  # Convert the WMO number to an integer

            # Find the parent folder dynamically
            parent_folder = None
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

            # Add the file path to the dictionary
            paths[station_id] = file_path

        return paths

    @property
    def _import(self):
        """module to import for to load this step in an Pipeline"""
        return "pyearthtools.tutorial"

def load(
        self,
        files: dict[str, Path] | Path | list[str | Path] | tuple[str | Path],
        combine: str = "by_coords",  # Default combine method
        **kwargs,
    ) -> Any:
        """
        Custom load method for HadISDIndex.

        Args:
            files (dict[str, Path] | Path | list[str | Path] | tuple[str | Path]):
                Files to load.
            combine (str, optional):
                Combine method for NetCDF files. Defaults to "by_coords".
                Options:
                    - "by_coords": Combine datasets by aligning coordinates.
                    - "nested": Combine datasets by concatenating along a new dimension.
            **kwargs:
                Additional arguments passed to the parent class's load method.

        Returns:
            Any:
                Loaded data.
        """
        # Pass the combine argument as part of **kwargs
        kwargs["combine"] = combine

        # Call the parent class's load method
        return super().load(files, **kwargs)

# Notes to Joel
# - Does PET have the ability to check a NetCDF file for the variables it contains?
# - If not, we should add that to PET so that a user can be given suggestions for what variables to select, should they give an incorrect variable name
# - # Convert querytime to Petdt for consistency. Useful for other classes that may not have the same conversion
       # querytime = Petdt(querytime)


# # New load method, not sure I like it:
#     def load(
#         self,
#         files: dict[str, Path] | Path | list[str | Path] | tuple[str | Path],
#         combine: str = "nested",  # Default to "nested" for station-specific data
#         **kwargs,
#     ) -> Any:
#         """
#         Custom load method for HadISDIndex.

#         Args:
#             files (dict[str, Path] | Path | list[str | Path] | tuple[str | Path]):
#                 Files to load.
#             combine (str, optional):
#                 Combine method for NetCDF files. Defaults to "nested".
#             **kwargs:
#                 Additional arguments passed to the parent class's load method.

#         Returns:
#             Any:
#                 Loaded data.
#         """
#         datasets = []

#         for station_id, file_path in files.items():
#             # Open the dataset
#             ds = xr.open_dataset(file_path, **kwargs)

#             # Add station-specific coordinates
#             ds = ds.assign_coords(
#                 {
#                     "station_id": station_id,
#                     "longitude": ds.longitude.values if "longitude" in ds else None,
#                     "latitude": ds.latitude.values if "latitude" in ds else None,
#                 }
#             )

#             # Drop conflicting global variables to avoid merge errors
#             ds = ds.drop_vars(["longitude", "latitude"], errors="ignore")
#             datasets.append(ds)

#         # Combine datasets along a new "station" dimension
#         combined_ds = xr.combine_nested(
#             datasets,
#             concat_dim="station",  # Concatenate along a new "station" dimension
#             combine_attrs="override",  # Handle conflicting attributes
#         )

#         return combined_ds