from __future__ import annotations

import functools
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Any, Literal
from dask.diagnostics import ProgressBar
from dask import delayed, compute

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
    "total_cloud_cover": -999.0,
    "low_cloud_cover": -999.0,
    "mid_cloud_cover": -999.0,
    "high_cloud_cover": -999.0,
}


@functools.lru_cache()
def cached_iterdir(path: Path) -> list[Path]:
    """Run iterdir but cached"""
    return list(path.iterdir())


@functools.lru_cache()
def cached_exists(path: Path) -> bool:
    """Run exits but cached"""
    return path.exists()


# TODO:
# - In the future it would be good to add the possibility to have this preprocessing step as part of a pipeline of other preprocessing steps.
# - Other similarly process heavy steps could be added to the pipeline, such as calculation of climatologies, or other derived variables.


# Helper function to preprocess and save NetCDF files as Zarr stores
# @delayed Experimenting with delayed to see if it helps with performance
def preprocess_and_save(file_path, date_range, zarr_output_dir):  # TODO Needs to be implemented correctly
    """
    Open a NetCDF file, preprocess it, and save as a Zarr store.

    Steps performed:
        - Opens the NetCDF file as an xarray Dataset.
        - Drops the 'input_station_id' variable if present (to avoid object dtype issues).
        - Assigns a 'station_id' coordinate from the dataset attributes or filename.
        - Reindexes the time dimension to a common hourly range.
        - Saves the processed Dataset to a Zarr store in the specified output directory.

    Args:
        file_path (str or Path): Path to the NetCDF file.
        date_range (tuple of str): (start, end) date strings for reindexing the time dimension.
        zarr_output_dir (str or Path): Directory where the Zarr store will be saved.

    Returns:
        str: Path to the saved Zarr store.
    """
    try:
        print(f"Preprocessing {file_path} -> {zarr_output_dir}")
        with xr.open_dataset(file_path) as ds:
            if "input_station_id" in ds:
                ds = ds.drop_vars("input_station_id")

            station_id = ds.attrs.get("station_id", file_path.stem)
            ds = ds.assign_coords(station_id=station_id)

            target_time = pd.date_range(date_range[0], date_range[1], freq="h")
            ds = ds.reindex(time=target_time)

            out_path = Path(zarr_output_dir) / f"{file_path.stem}.zarr"
            print(f"Saving to Zarr: {out_path}")
            ds.to_zarr(str(out_path), mode="w")
            print(f"Saved Zarr: {out_path}")
            return str(out_path)
    except Exception as e:
        print(f"Failed to preprocess {file_path}: {e}")
        raise


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
        variables: list[str] | str | None = None,
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
        base_transform += Drop("reporting_stats")

        # Add a transform to select variables (if variables are provided)
        if variables:
            base_transform += pyearthtools.data.transforms.variables.Select(self.variables)
            print(f"Variables selected: {self.variables}")

        # Possibly remove this transform if not needed
        base_transform += SetMissingToNaN(varname_val_map)

        if transforms is None:
            super().__init__(
                transforms=base_transform + TransformCollection(),
            )
        else:
            super().__init__(
                transforms=base_transform + transforms,
            )

        self.record_initialisation()

    # def get_all_station_ids(self, root_directory: Path | str) -> list[str]:
    #     """
    #     Retrieve all station IDs by scanning the dataset directory.

    #     Args:
    #         root_directory (Path | str): The root directory containing station data.

    #     Returns:
    #         list[str]: A list of all station IDs.
    #     """
    #     root_directory = Path(root_directory)
    #     station_ids = []
    #     for folder in cached_iterdir(root_directory):
    #         if folder.is_dir():
    #             for file in cached_iterdir(folder):
    #                 if file.suffix == ".nc":  # Check for NetCDF files
    #                     # Extract the station ID from the filename
    #                     station_id = file.stem.split("_")[-1]  # Assuming station ID is the last part of the filename
    #                     station_ids.append(station_id)
    #     return station_ids

    def get_all_station_ids(self, root_directory: Path | str = None) -> list[str]:
        """
        Retrieve all station IDs by scanning the dataset directory.

        Args:
            root_directory (Path | str, optional): The root directory containing station data.
                Defaults to HADISD_HOME/netcdf.

        Returns:
            list[str]: A list of all station IDs.
        """

        HADISD_HOME = self.ROOT_DIRECTORIES["hadisd"]
        if root_directory is None:
            # Search all WMO folders for netcdf subfolders
            wmo_folders = [f for f in Path(HADISD_HOME).iterdir() if f.is_dir() and f.name.startswith("WMO_")]
            station_ids = []
            for wmo_folder in wmo_folders:
                netcdf_dir = wmo_folder / "netcdf"
                if cached_exists(netcdf_dir):
                    for file in cached_iterdir(netcdf_dir):
                        if file.suffix == ".nc":
                            station_id = file.stem.split("_")[-1]
                            station_ids.append(station_id)
            return station_ids
        else:
            root_directory = Path(root_directory)
            if not cached_exists(root_directory):
                raise DataNotFoundError(f"Root directory does not exist: {root_directory}")
            station_ids = []
            for file in cached_iterdir(root_directory):
                if file.suffix == ".nc":
                    station_id = file.stem.split("_")[-1]
                    station_ids.append(station_id)
            return station_ids

    def filesystem(self, *args, date_range=("1970-01-01T00", "2023-12-31T23"), **kwargs) -> dict[str, Path]:
        """
        Map a station ID or list of station IDs to their corresponding file paths.

        Args:
            station_ids (str | list[str] | None): Station ID or list of station IDs. If None, use self.station.

        Returns:
            dict[str, Path]: A dictionary mapping station IDs to their corresponding file paths.

        Raises:
            DataNotFoundError: If a file is not found for any station ID.
        """

        HADISD_HOME = self.ROOT_DIRECTORIES["hadisd"]
        station_ids = self.station

        # Ensure station_ids is always a list
        if isinstance(station_ids, str):
            station_ids = [station_ids]

        # Retrieve all station IDs from the dataset directory if "all" is present
        if "all" in station_ids:
            station_ids = self.get_all_station_ids(HADISD_HOME)

        # Validate that station_ids is a list of strings
        if not isinstance(station_ids, list) or not all(isinstance(sid, str) for sid in station_ids):
            raise TypeError(f"Expected station_ids to be a str or list[str], but got: {type(station_ids)}")

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
            filename_nc = f"{version}_{date_range}_{station_id}.nc"
            filename_zarr = f"{version}_{date_range}_{station_id}.zarr"

            # Construct the full path
            file_path_nc = Path(HADISD_HOME) / parent_folder / "netcdf" / filename_nc
            file_path_zarr = Path(HADISD_HOME) / parent_folder / "zarr" / filename_zarr

            # Check if the file exists (comment out if testing with single netcdf)
            if not file_path_zarr.exists():
                raise DataNotFoundError(f"File not found for station: {station_id}, path: {file_path_zarr}")

            # Add the file path to the dictionary
            paths[station_id] = (
                file_path_zarr  # Change to file_path_zarr to test with zarr files or remove "_zarr" to test with netcdf files
            )

        return paths

    def load(
        self,
        files: dict[str, Path] | Path | list[str | Path] | tuple[str | Path],
        combine: str = "nested",
        concat_dim: str = "station",
        parallel: bool = True,
        # engine: Literal["netcdf4", "zarr"] = "zarr",  # Default engine for loading
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
        kwargs["concat_dim"] = concat_dim
        kwargs["parallel"] = parallel

        # Call the parent class's load method
        return super().load(files, **kwargs)

    @property
    def _import(self):
        """module to import for to load this step in a Pipeline"""
        return "pyearthtools.tutorial"
