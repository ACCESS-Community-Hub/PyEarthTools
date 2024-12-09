# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations
import json
import os
from typing import Any, Callable

import xarray as xr
import numpy as np
import pandas as pd

from pathlib import Path
from importlib.resources import files, as_file
import yaml
import copy


import pyearthtools.utils

pyearthtools_CATALOG_EXTENSIONS = [".cat", ".catalog"]
BLACKLISTED_EXTENSIONS = []
BLACKLISTED_EXTENSIONS.extend(pyearthtools_CATALOG_EXTENSIONS)


def filter_files(
    files: list[str | Path] | tuple[str | Path],
) -> list[str | Path] | tuple[str | Path]:
    """
    Filter blacklisted files out
    """
    new_files = []
    for file in files:
        if Path(file).suffix not in BLACKLISTED_EXTENSIONS:
            new_files.append(file)
    return new_files


def open_dataset(
    location: str | Path | tuple[str | Path] | list[str | Path],
    soft_fail: bool = False,
    **kwargs,
) -> xr.Dataset | tuple[xr.Dataset] | dict[str, Any]:
    """
    Open netcdf files at location using xarray

    If location is directory open all files inside

    Args:
        location (str | Path | tuple | list):
            Location to load data from. Can be folder, list of files, or single file.
        soft_fail (bool, optional):
            If set and cannot merge datasets together, return a dictionary of the data. Defaults to False.

    Returns:
        (xr.Dataset | tuple[xr.Dataset] | dict[str, xr.Dataset]):
            Loaded Datasets.
            If multiple files given, xarray fails to merge, and `soft_fail == True`,
             return a dictionary of {path difference: dataset}.
    """

    def get_config(mf: bool = False):
        open_kwargs = copy.copy(pyearthtools.utils.config.get("data.open.xarray"))
        if mf:
            open_kwargs.update(copy.copy(pyearthtools.utils.config.get("data.open.xarray_mf")))
        open_kwargs.update(kwargs)
        return open_kwargs

    if isinstance(location, (tuple, list)):
        try:
            return xr.open_mfdataset(filter_files(location), **get_config(True))

        except xr.MergeError as e:
            if not soft_fail:
                raise ValueError(
                    f"Cannot merge data from files: {filter_files(location)}.\nSet `soft_fail` to True to return a dictionary of the data from each of the sources, loaded separately."
                ) from e

            similar_path = Path(os.path.commonprefix(filter_files(location))).parent
            return {str(Path(d).relative_to(similar_path)): open_dataset(d) for d in filter_files(location)}

        except ValueError:
            # Try combining with 'nested'
            open_kwargs = get_config(True)
            open_kwargs["combine"] = "nested"
            # return xr.open_mfdataset(filter_files(location), **kwargs)
            return xr.open_mfdataset(
                filter_files(location),
            )

        except NotImplementedError:
            kwargs["chunks"] = None
            return open_dataset(location, **kwargs)

    path_location: Path
    path_location = Path(location)

    if path_location.is_dir():
        return open_dataset(list(path_location.iterdir()), **kwargs)

    try:
        preprocess = kwargs.pop("preprocess", lambda x: x)
        return preprocess(xr.open_dataset(location, **get_config(False)))
    except NotImplementedError:
        kwargs["chunks"] = None
        return open_dataset(location, **kwargs)


NETCDF_FILE_EXTENSONS = [".nc", ".netcdf"]
ZARR_FILE_EXTENSONS = [".zarr"]
NUMPY_EXTENSIONS = [".npy", ".np", ".numpy"]
CSV_EXTENSIONS = [".csv"]
JSON_EXTENSIONS = [".json"]

_REVERSED_LOADING_FUNCTIONS = {
    open_dataset: NETCDF_FILE_EXTENSONS,
    xr.open_zarr: ZARR_FILE_EXTENSONS,
    np.load: NUMPY_EXTENSIONS,
    pd.read_csv: CSV_EXTENSIONS,
    lambda x: json.load(open(x, "r")): JSON_EXTENSIONS,
    # pyearthtools.data.load: pyearthtools_CATALOG_EXTENSIONS,
}


def invert_dictionary_list(
    dictionary: dict[Callable, list[str]],
) -> dict[str, Callable]:
    return_dict = {}
    for key, value in dictionary.items():
        for item in value:
            return_dict[item] = key
    return return_dict


LOADING_FUNCTIONS = invert_dictionary_list(_REVERSED_LOADING_FUNCTIONS)


def check_extension(files: tuple[str | Path, ...] | list[str | Path]) -> list[str]:
    file_extens = []

    for filename in files:
        filename = Path(filename)
        extension = f".{filename.suffix.removeprefix('.')}"

        if extension not in LOADING_FUNCTIONS:
            if extension in BLACKLISTED_EXTENSIONS:
                continue

            raise KeyError(
                f"""Unable to load '{filename}' as extension: {extension!r} is not recognised.
                Recognised extensions include: {list(LOADING_FUNCTIONS.keys())}"""
            )

        if extension not in file_extens:
            file_extens.append(extension)

    return file_extens


def open_files(
    files: tuple[str | Path] | list[str | Path] | dict[str, str | Path] | str | Path,
    **kwargs,
) -> Any | tuple[Any]:
    """
    Open files from file list

    Supports different data formats, will automatically
    change how data is loaded from file extension.

    - numpy    ".npy"
    - csv      ".csv"
    - netcdf   ".nc"
    - json     ".json"

    If all files are netcdf, files are merged together

    Args:
        files (tuple | list | dict | str | Path):
            Files to load, if directory, load all files inside
        **kwargs (Any, optional):
            Keyword arguments to pass to loading function

    Returns:
        (Any | tuple[Any]):
            Opened Data, if only one found, return that one exactly
    """
    import pyearthtools.data

    global LOADING_FUNCTIONS
    if pyearthtools_CATALOG_EXTENSIONS[0] not in LOADING_FUNCTIONS:
        LOADING_FUNCTIONS.update(invert_dictionary_list({pyearthtools.data.load: pyearthtools_CATALOG_EXTENSIONS}))

    if isinstance(files, (str, Path)):
        if Path(files).is_dir() and not Path(files).suffix == ".zarr":
            files = [file for file in Path(files).glob("*") if file.is_file() and not file.name.startswith(".")]
        else:
            files = [files]

    if not isinstance(files, (tuple, list, dict)):
        raise TypeError(
            f"Could not load files from input: {files!r}.\n" "Supported types are (str, Path, tuple, list, dict)"
        )

    files_to_load = [value for _, value in files.items()] if isinstance(files, dict) else list(files)

    file_extens = check_extension(files_to_load)

    if len(file_extens) == 1 and file_extens[0] in NETCDF_FILE_EXTENSONS:
        return open_dataset(files_to_load, **kwargs)

    preprocess = kwargs.pop("preprocess", lambda x: x)

    opened_files = tuple(
        LOADING_FUNCTIONS[f".{Path(filename).suffix.removeprefix('.')}"](filename, **kwargs)
        for filename in files_to_load
    )
    opened_files = preprocess(opened_files)

    if isinstance(opened_files, (tuple, list)):
        if len(opened_files) == 1:
            return opened_files[0]
    return opened_files


def open_static(class_path: str | Path, file_path: str | Path | None = None) -> dict[str, Any] | list[str]:
    """Open file from class and file path

    If `file_path` not given it will be pulled as the last two '.' split items from
    `class_path`.

    Can open `yaml`, `.struc`, `.txt`, and `.valid` files,
    if yaml type, will return dictionary, if txt type will return list

    Args:
        class_path (str | Path):
            Class to locate file in
        file_path (str | Path, optional):
            Filename. Defaults to None.

    Raises:
        ValueError:
            If unable to load file

    Returns:
        (dict | list):
            Opened file
            If yaml type, will return dictionary, if txt type will return list
    """
    if file_path is None:
        file_path = ".".join(str(class_path).split(".")[-2:])
        class_path = ".".join(str(class_path).split(".")[:-2])

    variables = files(str(class_path)).joinpath(str(file_path))

    with as_file(variables) as file:
        if Path(file_path).suffix in [".yaml", ".yml", ".struc"]:
            return yaml.safe_load(open(file, "r"))

        elif Path(file_path).suffix in [".valid", ".txt"]:
            return open(file).read().splitlines()

    raise ValueError(f"Unable to load {class_path}.{file_path}")
