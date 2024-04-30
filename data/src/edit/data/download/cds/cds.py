# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

"""
Copernicus Data Storage Root Index
"""

from __future__ import annotations
from abc import abstractmethod, ABCMeta
from pathlib import Path
from typing import Iterable, Any, Type
import xarray as xr
import logging
import warnings

import urllib3

from edit.data import DataNotFoundError, EDITDatetime
from edit.data.download import DownloadIndex
from edit.data.indexes import utilities
from edit.data.patterns import TemporalExpandedDateVariable, PatternIndex
from edit.data.transform import Transform, TransformCollection

LOG = logging.getLogger("edit.data.download.cds")


def as_list(item: Any) -> list:
    """Get `item` as list"""
    if isinstance(item, (tuple, list)):
        return list(item)
    elif isinstance(item, str):
        return [item]
    elif isinstance(item, Iterable):
        return list(item)
    return [item]


class root_cds(DownloadIndex, metaclass=ABCMeta):
    """
    Root index to provide caching and access to cds.

    The child class must implement `_get_from_cds` to define the retrieval query.

    !!! Warning
        Currently this will download all data requested if only one is missing from disk
        And, if data was downloaded with a specific pressure level, requesting another will break.
    """

    def __init__(
        self,
        *,
        cache: str | Path | None = "temp",
        pattern: PatternIndex | Type | str = TemporalExpandedDateVariable,
        pattern_kwargs: dict[str, Any] = {},
        transforms: Transform | TransformCollection = TransformCollection(),
        download_transforms: Transform | TransformCollection = TransformCollection(),
        quiet: bool = True,
        **kwargs,
    ):
        """
        `edit.data` Index to download any data from Copernicus Data Store,

        The child class must implement `_get_from_cds` to define the retrieval query.

        Can set `cleanup` to limit `cache` directory size. See `edit.data.CachingIndex` for more.


        Args:
            cache (str | Path | None, optional):
                Location to cache data to, if not given will not cache. Defaults to 'temp''.
            pattern (PatternIndex | type | str, optional):
                Override for which pattern to use to save data. If str, relative to `edit.data.patterns.`.
                Defaults to `TemporalExpandedDateVariable`.
            pattern_kwargs (dict[str, Any], optional):
                Kwargs to pass to pattern init. Defaults to {}.
            transforms (Transform | TransformCollection, optional):
                Transforms to apply when retrieving data. Defaults to TransformCollection().
            download_transforms (Transform | TransformCollection, optional):
                Transforms to apply when downloading data. Defaults to TransformCollection().
            quiet (bool, optional):
                Whether to quiet all `cdsapi` info dumps. Defaults to True.

        Raises:
            ImportError:
                If `cdsapi` cannot be imported.
        """

        try:
            import cdsapi

            self._cds = cdsapi
            self._client = cdsapi.Client()

        except Exception as e:
            warnings.warn(
                f"Setting up cds raised the following error: {e}. \nWill not be able to download data.", RuntimeWarning
            )
            self._cds_error = e

        super().__init__(
            cache=cache,
            pattern=pattern if cache is not None else None,
            pattern_kwargs=pattern_kwargs if cache is not None else {},
            transforms=transforms,
            **kwargs,
        )

        self.quiet = quiet
        self.download_transforms = TransformCollection(download_transforms)

        if not hasattr(self, "catalog"):
            self.make_catalog()
        self._save_catalog(self.catalog)

    @abstractmethod
    def _get_from_cds(*args, **kwargs) -> tuple[str, dict] | list[tuple[str, dict]]:
        """
        Get Name and request dictionary to use when getting data from `cds`

        Can be list for multiple requests.

        !!! Note
            Must be implemented by child class
        """
        raise NotImplementedError(f"Child class must implement `_get_from_cds`.")

    def retrieve(self, *args, **kwargs):
        try:
            return super().retrieve(*args, **kwargs)

        except DataNotFoundError as e:
            LOG.warn(f"Attempting to generate data again due to {e.message}")

        with self.override:
            return super().retrieve(*args, **kwargs)

    def download(self, *args, **kwargs) -> xr.Dataset:
        """
        Download data from Copernicus

        The child class must implement `_get_from_cds`, and return
            (
                NAME = Name to pass to `cdsapi.Client().retrieve`,
                REQUEST = Request dict to pass to `cdsapi.Client().retrieve`,
            )

        Any data downloaded is temporary until saved.

        Raises:
            RuntimeError:
                If no internet connection can be established.
            TypeError:
                If cannot parse `_get_from_cds` or cds.retrieve result.

        Returns:
            (xr.Dataset):
                Downloaded data.
        """
        if not hasattr(self, "_cds"):
            raise type(self._cds_error)(f"Setting up cds raised this error: {self._cds_error}")

        cds_request = self._get_from_cds(*args, **kwargs)

        if not isinstance(cds_request, (tuple, list)):
            raise TypeError(f"Cannot parse result from `_get_from_cds` of {type(cds_request)}")

        def retrieve(request: tuple[str, dict]):
            """Actually run cds retrieval"""
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            if self.quiet:
                logging.getLogger("urllib3").setLevel(logging.ERROR)
                logging.getLogger("cdsapi").setLevel(logging.WARN)

            result = None
            excep = None

            try:
                result = self._client.retrieve(*request)
            except Exception as e:
                excep = e

            if result is None:
                raise DataNotFoundError(
                    f"Data could not be retrieved from {self.__class__.__qualname__} for {args}, {kwargs}"
                ) from excep

            if not isinstance(result, self._cds.api.Result):
                raise TypeError(
                    f"Cannot parse result of type {type(result)}. Pass either str, Path or `cdsapi.api.Result`"
                )
            path = self.get_tempdir() / "download.nc"

            # with warnings.catch_warnings():
            #     warnings.filterwarnings(action = 'ignore' if self.quiet else 'once', category=urllib3.exceptions)
            LOG.info(f"Beginning download from cds.")
            result.download(path)

            return path

        if isinstance(cds_request, list):  # Map multiple queries
            return self.download_transforms(utilities.open_files(list(map(retrieve, cds_request))))
        return self.download_transforms(utilities.open_files(retrieve(cds_request)))


class cds(root_cds):
    """
    General cds retreival

    Allows a user to specify all options to be retrieved from cds, with the time auto handled.
    """

    @property
    def _desc_(self):
        return {
            "singleline": "Copernicus Data Store ",
            "Documentation": "https://cds.climate.copernicus.eu/",
        }

    def __init__(self, cds_dataset: str, cds_kwargs: dict[str, Any] = {}, **kwargs: dict[str, Any]):
        """
        General cds downloader

        Allows a user to specify all options to be retrieved from cds, with the time auto handled.

        See `cds` for kwarg explanation.

        Args:
            cds_dataset (str):
                CDS dataset to retrieve from
            cds_kwargs (dict[str, Any], optional):
                Kwargs to pass to cds request. Defaults to {}.
        """

        self.cds_dataset = cds_dataset
        self.cds_kwargs = cds_kwargs

        if "variables" in cds_kwargs:
            kwargs["pattern_kwargs"] = kwargs.pop("pattern_kwargs", {})
            kwargs["pattern_kwargs"]["variables"] = as_list(kwargs["pattern_kwargs"].pop("variable", []))

            kwargs["pattern_kwargs"]["variables"].extend(as_list(cds_kwargs["variables"]))

        super().__init__(**kwargs) # type: ignore

    def _get_from_cds(self, querytime: EDITDatetime | str):
        querytime = EDITDatetime(querytime)

        base_dict = {
            "format": "netcdf",
            "year": querytime.year,
            "month": querytime.month,
            "day": querytime.day,
            "time": querytime.strftime("%H:%M"),
        }
        base_dict.update(self.cds_kwargs)
        return self.cds_dataset, base_dict
