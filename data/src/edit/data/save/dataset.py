# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import xarray as xr

from edit.data.indexes import FileSystemIndex
from edit.data.save.utils import ManageFiles

VALID_EXTENSIONS = [".nc", ".netcdf"]
DATASET_TIMEOUT = 60


def save(
    dataset: tuple[xr.Dataset] | xr.DataArray | xr.Dataset,
    callback: FileSystemIndex,
    *args,
    save_kwargs: dict[str, Any] = {},
    try_thread_safe: bool = True,
    **kwargs,
):
    """
    Saves a dataset based on a callback to an index.

    Supports:
        dataset: xr.Dataset, xr.DataArray, tuple of either
        callback.search(): Path, str, or dictionary of either
            If dict, will only save dataset, and will only save specified keys

    """

    callback_paths = callback.search(*args, **kwargs)

    if isinstance(callback_paths, dict):
        if not isinstance(dataset, xr.Dataset):
            raise TypeError(f"A pattern returning a dictionary, can only save datasets, not {type(dataset)}")
        tuple(map(lambda x: Path(x).parent.mkdir(parents=True, exist_ok=True), callback_paths.values()))

        subset_paths = {
            key: Path(callback_paths[key]) for key in set(dataset.data_vars).intersection(callback_paths.keys())
        }
        if len(subset_paths.keys()) < len(dataset.data_vars):
            warnings.warn(
                "Some data variables are missing a save path, and will not be saved.\n"
                f"{set(dataset.data_vars).difference(subset_paths.keys())}",
                UserWarning,
            )

        with ManageFiles(
            list(subset_paths.values()), timeout=DATASET_TIMEOUT, lock=try_thread_safe, uuid=not try_thread_safe
        ) as (temp_files, exist):
            if not exist:
                xr.save_mfdataset(tuple(dataset[[var]] for var in subset_paths.keys()), temp_files, **save_kwargs)

        return subset_paths

    if not isinstance(callback_paths, (str, Path)):
        raise TypeError(f"Cannot parse 'paths' of type {type(callback_paths)!r}")

    path = Path(callback_paths)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix not in VALID_EXTENSIONS:
        raise ValueError(
            f"Saving netcdf files requires a suffix in {VALID_EXTENSIONS}, not {path.suffix!r} on {path!r}"
        )

    if isinstance(dataset, (tuple, list)):
        for i, data in enumerate(dataset):
            if isinstance(data, xr.DataArray):
                data = data.to_dataset(name="data")

            subpath = (path / f"{i}").with_suffix(path.suffix)
            subpath.parent.mkdir(parents=True, exist_ok=True)

            with ManageFiles(subpath, timeout=DATASET_TIMEOUT, lock=try_thread_safe, uuid=not try_thread_safe) as (
                temp_file,
                exist,
            ):
                if not exist:
                    assert isinstance(temp_file, (str, Path))
                    data.to_netcdf(temp_file, **save_kwargs)
    else:
        if isinstance(dataset, xr.DataArray):
            dataset = dataset.to_dataset(name="data")

        with ManageFiles(path, timeout=DATASET_TIMEOUT, lock=try_thread_safe, uuid=not try_thread_safe) as (
            temp_file,
            exist,
        ):
            if not exist:
                assert isinstance(temp_file, (str, Path))
                dataset.to_netcdf(temp_file, **save_kwargs)

    return callback_paths
