# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
zarr Archives

Allows access and saving in zarr
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from os import PathLike

import xarray as xr
import dask
import logging

import pyearthtools.data
from pyearthtools.data.time import pyearthtoolsDatetime
from pyearthtools.data.transforms import Transform, TransformCollection
from pyearthtools.data.indexes.indexes import DataFileSystemIndex, TimeIndex

from pyearthtools.data.indexes.utilities.dimensions import identify_time_dimension
from pyearthtools.data.utils import parse_path

from pyearthtools.data.save import save

LOG = logging.getLogger("pyearthtools.data")


class ZarrIndex(DataFileSystemIndex):
    """
    Zarr Index

    Can be used to access local/remote zarr archives, with the ability to write into them.

    ## Example:
    ```python
    zarr_archive = Zarr(PATH_TO_ZARR_ARCHIVE)
    zarr_archive()
    ```

    For time aware indexing, use `ZarrTime`.

    Additonally, this class can be used to create an 'empty' archive, with all metadata prepopulated.

    This is useful to premake an archive, and then use many distributed processes to write subsets into it.

    ## Template Example:
    ```python
    zarr_archive = Zarr(PATH_TO_ZARR_ARCHIVE, template = True)
    zarr_archive.make_template(SINGLE_SAMPLE, time = EXPANDED_TIME)

    for subsample in TOTALSAMPLES: # Can be done distributedly
        zarr_archive.save(subsample)
    ```
    """

    @property
    def _desc_(self):
        return {"singleline": f"Zarr Archive at {str(self._store)!r}."}

    def __init__(
        self,
        store: PathLike,
        variables: str | list[str] | None = None,
        *,
        template: bool = False,
        transforms: Transform | TransformCollection | None = None,
        open_kwargs: dict[str, Any] | None = None,
        save_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        Zarr Archive

        Can use `sa` as mode for saving, which means 'safe append'.
        Will look at existing archive, and only append on `append_dim` data that is missing.

        If `template` is True, `exists` will always be False.

        Args:
            store (PathLike):
                Store or path to directory in local or remote file system.
            variables (str | list[str] | None, optional):
                Variables within the dataset to subset to. Defaults to None.
            template (bool, optional):
                Whether this archive is a template, will cause `exists` to always return False.
                Allows a cacher to write to this archive, despite it appearing to exist on disk.
                Defaults to False.
            transforms (Transform | TransformCollection | None, optional):
                Base Transforms to be applied to data. Transforms are applied on the retrieval
                of data, i.e. `index[]` but not when directly getting the data, `index.get()`.
                Defaults to TransformCollection().
            open_kwargs (dict[str, Any] | None, optional):
                Kwargs to use when opening the zarr archive.
                See https://docs.xarray.dev/en/stable/generated/xarray.open_zarr.html
                Defaults to None.
            save_kwargs (dict[str, Any] | None, optional):
                Kwargs to use when saving the zarr archive.
                See https://docs.xarray.dev/en/latest/generated/xarray.Dataset.to_zarr.html
                Defaults to None.
        """

        super().__init__(transforms=transforms or TransformCollection(), **kwargs)
        self.record_initialisation()

        self._store = parse_path(store)

        self._variables = variables
        self._template = template

        if template and not self._store.parent.exists():
            self._store.parent.mkdir(exist_ok=True, parents=True)

        self._open_kwargs = open_kwargs or {}
        self._save_kwargs = save_kwargs or {}

    def get(self):
        """
        Get zarr archive

        Used within the indexes data access flows

        Subset on `variables` if given, but applies no other subsetting.
        """
        base_zarr = self._get_zarr()
        if self._variables is not None:
            var_transform = pyearthtools.data.transform.variables.Trim(self._variables)
            base_zarr = var_transform(base_zarr)
        return base_zarr

    def save(self, data: xr.Dataset, save_kwargs: dict[str, Any] | None = None, **kwargs):
        """
        Save `data` into the zarr archive

        See https://docs.xarray.dev/en/latest/generated/xarray.Dataset.to_zarr.html

        Can use `sa` as mode for saving, which means 'safe append'.
        Will look at existing archive, and only append on `append_dim` data that is missing.

        Args:
            data (xr.Dataset):
                Dataset to save
            save_kwargs (dict[str, Any] | None, optional):
                Extra kwargs to pass to `.to_zarr`, in addition to `init.save_kwargs`. Defaults to None.
            **kwargs:
                Kwargs form of `save_kwargs`s
        """
        skwargs = dict(self._save_kwargs)
        skwargs.update(save_kwargs or {})
        skwargs.update(kwargs)

        if self._store.exists() and self._template:
            skwargs["region"] = skwargs.pop("region", "auto")

        try:
            save(data, self, save_kwargs=skwargs, zarr=True)
        except Exception as e:
            if not self._template:
                raise ValueError(
                    f"Could not save to {self._store!s}, if you ware attempting to save to a template, ensure `template == True` in init."
                ) from e
            raise e

    def _get_zarr(self) -> xr.Dataset:
        """Open zarr archive"""
        return xr.open_zarr(self._store, **self._open_kwargs)

    def make_template(
        self,
        dataset: xr.Dataset,
        *,
        chunk: Literal["auto"] | None | dict[str, Literal["auto"] | int] = None,
        encoding: dict[str, dict[str, Any]] | None = None,
        overwrite: bool = True,
        append_dimension: str | None = None,
        expand_coords: dict[str, list[Any]] | None = None,
        **kwargs,
    ):
        """
        Make a template dataset out of one sample of data, `dataset`.

        A sample should contain all of the variables this full dataset should have.
        It must also contain all values along the coordinates not included in
        `expand_coords` that can be expected, i.e. all latitude values.

        A sample does not need to include all values as specified in `expand_coords`,
        it will be reindexed to include them by this function.

        The full dataset is defined as the sample expanded by `expand_coords`.

        Args:
            dataset (xr.Dataset):
                Single sample of full dataset.
                All metadata will be taken from this sample.
            chunk (Literal['auto'] | None | dict[str, Literal['auto']  |  int ], optional):
                Override for chunks of zarr archive.
                Any key in `expand_coords` will be chunked 'auto'. Defaults to None.
            overwrite (bool, optional):
                Whether to override an existing zarr archive. Defaults to True.
            append_dimension (str | None, optional):
                Dimension to append on, if to append. Defaults to None.
            expand_coords (dict[str, list[Any]] | None):
                Coordinates to reindex. Allows for a single sample to be passed,
                but full archive created of larger data.
                Defaults to None.
            kwargs:
                Kwargs form of `expand_coords`

        Raises:
            FileExistsError:
                If file exists and `override` == False.

        Examples:
        ```python
        era5 = pyearthtools.data.archive.ERA5.sample()

        full_time_values = list(map(lambda x: x.datetime64(), pyearthtools.data.TimeRange('1980', '2020', '6 hour')))

        zarr_archive = Zarr(PATH_TO_ZARR, template = True)
        zarr_archive.make_template(era5['2000-01-01T00'], time = full_time_values)

        # Will create a zarr archive like `era5` but across all of `full_time_values`
        ```
        """
        expand_coords = dict(expand_coords or {})
        expand_coords.update(kwargs)

        with dask.config.set(**{"array.slicing.split_large_chunks": False}):  # type: ignore
            for key, val in expand_coords.items():
                enc = dataset[key].encoding
                dataset = dataset.reindex({key: val}).chunk({key: "auto"})
                dataset[key].encoding.update(enc)

            if chunk is not None:
                dataset = dataset.chunk(chunk)

            if encoding is not None:
                dataset = pyearthtools.data.transforms.attributes.SetEncoding(encoding)(dataset)

            save_kwargs = {}

            if Path(self._store).exists():
                if not overwrite:
                    if append_dimension is None:
                        raise FileExistsError(
                            f"{self._store!s} already exists and `overwrite` is False, and no `append_dimension` is given."
                        )
                    save_kwargs["mode"] = "a"
                    save_kwargs["append_dim"] = append_dimension
                else:
                    save_kwargs["mode"] = "w"

            save(dataset, self, save_kwargs=dict(compute=False, **save_kwargs), zarr=True)

    def exists(self, search_dict: dict[str, Any] | None = None, **kwargs) -> bool:
        """
        Check if zarr archive exists

        If `template == True`, always return False,

        Args:
            search_dict (dict[str, Any] | None, optional):
                Key / val to check for in data. Defaults to None.
            kwargs:
                Kwargs form of `search_dict`.

        Returns:
            (bool):
                If zarr archive / data in archive exists.
        """
        search_dict = dict(search_dict or {})
        search_dict.update(kwargs)

        if self._template:
            return False

        if not Path(self._store).exists():
            LOG.debug(f"Exists check failed as path {self._store} does not exist.")
            return False

        try:
            zarr = self._get_zarr()
        except FileNotFoundError:
            return False

        exists_bool = True

        for key, val in search_dict.items():
            exists_bool = exists_bool and val in zarr[key]

            if not exists_bool:
                return exists_bool

        return exists_bool

    def search(self) -> str | Path:
        """Get path of zarr archive"""
        return self._store


class ZarrTimeIndex(ZarrIndex, TimeIndex):
    """
    Time index aware zarr archive

    Allows for `[]` with a time value, and subsetting accordingly.
    """

    def retrieve(
        self,
        querytime: str | pyearthtoolsDatetime | None = None,
        *args,
        transforms: Transform | TransformCollection | None = None,
        **kwargs,
    ) -> Any:
        """
        If supplied, retrieve the data subset for the specified time
        """
        base_data = super().retrieve(*args, transforms=transforms or TransformCollection(), **kwargs)
        if querytime is not None:

            def to_np(x):
                return pyearthtoolsDatetime(x).datetime64()

            base_data = base_data.sel(
                {
                    identify_time_dimension(base_data): (
                        [to_np(querytime)] if not isinstance(querytime, list) else list(map(to_np, querytime))
                    )
                }
            )
        if identify_time_dimension(base_data) in base_data:
            base_data = base_data.sortby(identify_time_dimension(base_data))
        return base_data

    def exists(self, querytime: str | None = None, **kwargs):
        """
        Check for existence,

        If `querytime` given check it is in the zarr archive.
        """
        exists_bool = super().exists(**kwargs)

        if not exists_bool:
            return exists_bool

        zarr = self._get_zarr()
        if querytime is not None:
            time_dim = identify_time_dimension(zarr)

            exists_bool = exists_bool and pyearthtoolsDatetime(querytime).datetime64() in zarr[time_dim]
        return exists_bool
