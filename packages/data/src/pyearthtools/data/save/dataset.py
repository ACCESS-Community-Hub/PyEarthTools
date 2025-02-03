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

from pathlib import Path
from typing import Any
import warnings

import xarray as xr

import pyearthtools.utils
from pyearthtools.data.indexes import FileSystemIndex
from pyearthtools.data.save.utils import ManageFiles

VALID_EXTENSIONS = [".nc", ".netcdf"]
DATASET_TIMEOUT = 60


def save(dataset, callback, *args, zarr: bool | None = None, save_kwargs: dict[str, Any], **kwargs):

    if zarr is None:
        path = callback.search(*args, **kwargs)
        if isinstance(path, (str, Path)) and Path(path).suffix == ".zarr":
            zarr = True
        else:
            zarr = False

    if not zarr:
        return to_netcdf(dataset, callback, *args, save_kwargs=save_kwargs, **kwargs)

    return to_zarr(dataset, callback, *args, save_kwargs=save_kwargs, **kwargs)


def to_netcdf(
    dataset: tuple[xr.Dataset] | xr.DataArray | xr.Dataset,
    callback: FileSystemIndex,
    *args,
    save_kwargs: dict[str, Any] | None = None,
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

    save_kwargs_default: dict = dict(pyearthtools.utils.config.get("data.save.xarray"))
    save_kwargs_default.update(save_kwargs or {})
    save_kwargs = save_kwargs_default

    if isinstance(callback_paths, dict):
        if not isinstance(dataset, xr.Dataset):
            raise TypeError(f"A pattern returning a dictionary, can only save datasets, not {type(dataset)}")
        tuple(
            map(
                lambda x: Path(x).parent.mkdir(parents=True, exist_ok=True),
                callback_paths.values(),
            )
        )

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
            list(subset_paths.values()),
            timeout=DATASET_TIMEOUT,
            lock=try_thread_safe,
            uuid=not try_thread_safe,
        ) as (temp_files, exist):
            if not exist:
                xr.save_mfdataset(
                    tuple(dataset[[var]] for var in subset_paths.keys()),
                    temp_files,
                    **save_kwargs,
                )

        return subset_paths

    if isinstance(callback_paths, (tuple, list)) and isinstance(dataset, (list, tuple)):
        if len(callback_paths) != len(dataset):
            raise ValueError(f"Lengths differ between paths and data. {len(callback_paths)} != {len(dataset)}.")

        with ManageFiles(
            callback_paths,
            timeout=DATASET_TIMEOUT,
            lock=try_thread_safe,
            uuid=not try_thread_safe,
        ) as (
            temp_files,
            exist,
        ):
            if not exist:
                xr.save_mfdataset(dataset, temp_files, **save_kwargs)
        return callback_paths

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

            with ManageFiles(
                subpath,
                timeout=DATASET_TIMEOUT,
                lock=try_thread_safe,
                uuid=not try_thread_safe,
            ) as (
                temp_file,
                exist,
            ):
                if not exist:
                    assert isinstance(temp_file, (str, Path))
                    data.to_netcdf(temp_file, **save_kwargs)
    else:
        if isinstance(dataset, xr.DataArray):
            dataset = dataset.to_dataset(name="data")

        with ManageFiles(
            path,
            timeout=DATASET_TIMEOUT,
            lock=try_thread_safe,
            uuid=not try_thread_safe,
        ) as (
            temp_file,
            exist,
        ):
            if not exist:
                assert isinstance(temp_file, (str, Path))
                dataset.to_netcdf(temp_file, **save_kwargs)

    return callback_paths


def to_zarr(
    dataset: xr.DataArray | xr.Dataset,
    callback: FileSystemIndex,
    *args,
    save_kwargs: dict[str, Any] | None = None,
    **kwargs,
):
    if isinstance(dataset, tuple):
        raise TypeError()

    save_kwargs_default: dict = dict(pyearthtools.utils.config.get("data.save.zarr"))
    save_kwargs_default.update(save_kwargs or {})
    save_kwargs = save_kwargs_default

    zarr_file = callback.search(*args, **kwargs)

    if not isinstance(zarr_file, (str, Path)):
        raise TypeError(f"Cannot save zarr file to {type(zarr_file)}, must be str or Path.")

    if "mode" in save_kwargs and save_kwargs["mode"] == "sa":
        if "append_dim" not in save_kwargs:
            raise ValueError("For safe append ('sa'), an `append_dim` must be specified.")

        save_kwargs["mode"] = "a"
        if callback.exists(*args, **kwargs):
            append_dim = save_kwargs["append_dim"]
            saved_zarr = xr.open_zarr(zarr_file)
            existing_dim = saved_zarr[append_dim]
            to_save_dim = dataset[append_dim]

            dataset = dataset.sel({append_dim: to_save_dim.isin(existing_dim) == False})  # noqa: E712
        else:
            save_kwargs.pop("mode")

    if Path.is_dir(zarr_file) and not Path(zarr_file).exists():
        save_kwargs.pop("append_dim", None)
        save_kwargs.pop("region", None)

    # print(save_kwargs)
    # print(dataset)
    dataset.to_zarr(zarr_file, **save_kwargs)
