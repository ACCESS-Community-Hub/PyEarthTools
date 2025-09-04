from __future__ import annotations

import functools
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Any

import pyearthtools.data
from pyearthtools.data.archive import register_archive
from pyearthtools.data.exceptions import DataNotFoundError
from pyearthtools.data.indexes import ArchiveIndex
from pyearthtools.data.transforms import Transform, TransformCollection
from pyearthtools.data.transforms.variables import Drop
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

    def get_all_station_ids(self, root_directory: Path | str = None) -> list[str]:
        """
        Retrieve all station IDs by scanning the Zarr directory.

        Args:
            root_directory (Path | str, optional): The directory containing Zarr files.
                Defaults to HADISD_HOME/zarr.

        Returns:
            list[str]: A list of all station IDs.
        """
        HADISD_HOME = self.ROOT_DIRECTORIES["hadisd"]
        if root_directory is None:
            zarr_dir = Path(HADISD_HOME) / "zarr"
        else:
            zarr_dir = Path(root_directory)

        if not cached_exists(zarr_dir):
            raise DataNotFoundError(f"Zarr directory does not exist: {zarr_dir}")

        station_ids = []
        for file in cached_iterdir(zarr_dir):
            if file.suffix == ".zarr":
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

        # Map station IDs to their file paths
        paths = {}
        for station_id in station_ids:
            date_range_str = "19310101-20240101"  # Hardcoded for now; adjust if dataset is updated
            version = "hadisd.3.4.0.2023f"
            filename_zarr = f"{version}_{date_range_str}_{station_id}.zarr"

            # filename_nc = f"{version}_{date_range_str}_{station_id}.nc" # Uncomment to test with netcdf
            # file_path_nc = Path(HADISD_HOME) / "netcdf" / filename_nc   # Uncomment to test with netcdf

            # Construct the full path
            file_path_zarr = Path(HADISD_HOME) / "zarr" / filename_zarr

            # Check if the file exists (comment out if testing with single netcdf)
            if not file_path_zarr.exists():
                raise DataNotFoundError(f"File not found for station: {station_id}, path: {file_path_zarr}")

            # Add the file path to the dictionary
            paths[station_id] = (
                file_path_zarr  # Change to file_path_zarr to test with zarr files or remove "_zarr" to test with netcdf files
                # file_path_nc  # Uncomment to test with netcdf files
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
