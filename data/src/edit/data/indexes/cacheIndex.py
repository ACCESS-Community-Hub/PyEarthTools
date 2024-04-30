# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Any, Literal
import warnings
import logging

import xarray as xr
from multiprocessing import Process

import edit.data

from edit.data import patterns, EDITDatetime, TimeDelta, CatalogEntry, DataNotFoundError
from edit.data.transform import Transform, TransformCollection
from edit.data.patterns.default import PatternIndex
from edit.data.indexes import ArchiveIndex, DataFileSystemIndex, ForecastIndex, TimeIndex
from edit.data.indexes.utilities.delete_files import delete_older_than, delete_path
from edit.data.indexes.utilities.folder_size import ByteSize, FolderSize

from edit.utils.context import ChangeValue

LOG = logging.getLogger(__name__)


class BaseCacheIndex(DataFileSystemIndex):
    """
    DataIndex Object that has no data on disk intially,
    but is being generated from other sources and saved in given cache.

    A child must implement the [_generate][edit.data.indexes.cacheIndex.CachingIndex._generate] function

    ## Data Flowchart
    ``` mermaid
        graph LR
        A[Data Request `.get`] --> B{Cache Given?};
        B -->| Yes| C{Data Exists...};
        C --> |No| G;
        C --> |Yes| D[Get Data from Cache];
        B --> |No| G[Generate Data];
    ```
    """

    _override: bool = False
    _cleanup: dict[str, Any] | float | int | str | None = None

    def __init__(
        self,
        cache: str | Path,
        pattern: str | type | PatternIndex | None = None,
        pattern_kwargs: dict[str, Any] | str = {},
        *,
        transforms: Transform | TransformCollection = TransformCollection(),
        cleanup: dict[str, Any] | float | int | str | None = None,
        override: bool | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Base DataIndex Object to Cache data on the fly

        If only `cache` is given, ExpandedDate, or TemporalExpandedDate will be used by default. If `cache` and `pattern` not given,
        will not save data, and the point of this class is lost.

        `cache` can also be 'temp' to set to a TemporaryDirectory created on `__init__`, or include any environment variables,
        with $NOTATION.

        !!! Existing Cache:
            If the `cache` is set to an existing cache location, and the `pattern` is the same being made and exists,
            `pattern_kwargs` will be set by default to the existing cache's kwargs, and then updated by any given.

        Args:
            cache (str | Path):
                Location to save data to.
            pattern (str | type | PatternIndex, optional):
                String of pattern to use or defined pattern. In order to save as catalog, a str is needed.
                Defaults to ExpandedDate, or TemporalExpandedDate.
            pattern_kwargs (dict, optional):
                Kwargs to pass to initalisation of new pattern if pattern is str. Defaults to {}.
            transforms (Transform | TransformCollection, optional):
                Base Transforms to apply. Defaults to TransformCollection().
            cleanup (dict | float | int | str | None, optional):
                Cache cleanup settings.
                If a number type, assumed to represent age of file in days
                If dictionary type, the following keys can be used:
                | Key | Purpose | Type |
                | --- | ------- | ---- |
                | delta | Time delta to delete files past | int, float, tuple, TimeDelta |
                | dir_size | Maximum allowed directory size. Deletes oldest according to `key` | int, float, str, ByteSize (if str, use '100 GB' format) |
                | key | Key to use to find time of file for other time based delete steps. Default 'modified'. | Literal['modified', 'created'] |
                | data_time | Maximum difference in time the data is of and current time | int, float, tuple, TimeDelta |
                | verbose | Print files being deleted | bool] |

                Cleanup is run on each initialisation and deletion of the `CacheIndex`, and can be triggered manually with `.cleanup()`

                Defaults to None.

        Raises:
            ValueError: If `cache` and `pattern` not given.
        """
        base_transform = TransformCollection() + transforms
        super().__init__(transforms=base_transform, **kwargs)

        if isinstance(pattern_kwargs, str):
            try:
                import json

                pattern_kwargs = json.loads(pattern_kwargs)
            except Exception as e:
                raise ValueError(f"Something went wrong parsing `pattern_kwargs`: {pattern_kwargs!r} to dict.") from e

        if not isinstance(pattern_kwargs, dict):
            raise TypeError(
                f"Cannot parse `pattern_kwargs`, must be a dictionary, or string in json form, not {pattern_kwargs!r}"
            )

        self._input_cache = cache
        self.pattern_kwargs = pattern_kwargs
        self.pattern_type = pattern

        self._cleanup = cleanup
        self._verbose = verbose

        self.make_catalog(name=str(self.cache) + "_Cache")

        if "data_interval" in kwargs:
            self.pattern_kwargs["data_interval"] = kwargs["data_interval"]

        if override is not None:
            warnings.warn("Override is deprecated, use `.override` for context manager", DeprecationWarning)

        if self._input_cache is None and self.pattern_type is None:
            warnings.warn(f"Without a `cache` nor `pattern` given, this `CachingIndex` will not cache.", UserWarning)

        _ = self.pattern

        self._save_catalog()
        Process(target=self.cleanup).run()

    @property
    def cache(self):
        if self._input_cache is None:
            return None
        return self.pattern.root_dir

    @cached_property
    def pattern(self) -> PatternIndex:
        """Get Pattern from `__init__` args"""

        if self._input_cache is None and self.pattern_type is None:
            raise AttributeError("`cache` nor `pattern` were provided on `init`, so can Pattern can be found.")

        pattern_kwargs = dict(self.pattern_kwargs)

        def _update_kwargs(spec_pattern: type, **kwargs: Any) -> dict[str, Any]:
            if self._input_cache is None or not Path(self._input_cache).exists():
                return kwargs
            try:
                loaded_catalog = edit.data.load(self._input_cache)
            except (ValueError, FileNotFoundError):
                return kwargs

            if not isinstance(loaded_catalog, PatternIndex):
                return kwargs

            if type(loaded_catalog) == spec_pattern:
                _kwargs = loaded_catalog.catalog._kwargs
                _kwargs.update(**kwargs)
                _kwargs.pop("root_dir")
                return _kwargs
            return kwargs

        if self._input_cache is not None and self.pattern_type is None:
            if "data_interval" in pattern_kwargs:
                pattern_index = patterns.TemporalExpandedDate
                pattern_kwargs["file_resolution"] = pattern_kwargs.pop(
                    "file_resolution", TimeDelta(pattern_kwargs["data_interval"]).resolution
                )

                if "directory_resolution" not in pattern_kwargs:
                    try:
                        pattern_kwargs["directory_resolution"] = (
                            TimeDelta(pattern_kwargs["data_interval"]).resolution - 1
                        )
                    except Exception:
                        pass
            else:
                pattern_index = patterns.ExpandedDate
            pattern_kwargs = _update_kwargs(pattern_index, **pattern_kwargs)

            return pattern_index(root_dir=self._input_cache, **pattern_kwargs)

        if isinstance(self.pattern_type, str):
            if self._input_cache is None:
                raise ValueError(f"With 'pattern' as a str, and no 'cache'. Location of data is unclear.")

            pattern_kwargs = _update_kwargs(getattr(patterns, self.pattern_type), **pattern_kwargs)
            return getattr(patterns, self.pattern_type)(root_dir=self._input_cache, **pattern_kwargs)

        elif isinstance(self.pattern_type, PatternIndex):
            self._input_cache = self.pattern_type.root_dir
            return self.pattern_type

        elif isinstance(self.pattern_type, type):
            pattern_kwargs = _update_kwargs(self.pattern_type, **pattern_kwargs)

            return self.pattern_type(root_dir=self._input_cache, **pattern_kwargs)

        else:
            raise TypeError(f"Cannot parse `pattern_type` of {type(self.pattern_type)}")

    def cleanup(self, complete: bool = False):
        """
        Cleanup cache directory using `cleanup` as provided in `__init__`.

        Args:
            complete (bool, optional):
                Complete directory cleanup.
                If set to True, this will delete all data in the cache.
                Defaults to False.
        """
        if complete and self.cache is not None:
            warnings.warn(f"Deleting all data in the cache at '{self.cache}'", UserWarning)
            self.__run_cleanup(delta=0)
            delete_path(self.cache)
            return

        if self._cleanup is None or self.cache is None:
            return

        if isinstance(self._cleanup, dict):
            self.__run_cleanup(**self._cleanup)
        else:
            if isinstance(self._cleanup, str) and "," in self._cleanup:
                if len(self._cleanup.split(",")) == 2:
                    split = self._cleanup.split(",")
                    self.__run_cleanup(**{split[0]: split[1]})
                    return
            try:
                self.__run_cleanup(delta=self._cleanup)
            except Exception as e:
                self.__run_cleanup(dir_size=self._cleanup)

    def __run_cleanup(
        self,
        delta: TimeDelta | int | float | tuple | None = None,
        dir_size: ByteSize | str | int | float | None = None,
        data_time=None,
        key: Literal["modified", "created"] = "modified",
        verbose: bool = True,
    ):
        """Run cleanup on cache

        Args:
            delta (TimeDelta | int | float | tuple | None, optional):
                Max time since `key`. Defaults to None.
            dir_size (ByteSize | str | int | float | None, optional):
                Maximum directory size. Defaults to None.
            data_time (_type_, optional):
                Max time since valid data. NOT IMPLEMENTED. Defaults to None.
            key (Literal['modified', 'created'], optional):
                Key to get of file, for use with delta. Defaults to "modified".
            verbose (bool, optional):
                Whether to list files being deleted.. Defaults to False.

        Raises:
            TypeError:
                If cache not specified for this Index
            ValueError:
                If no cleanup args given
        """

        if self.cache is None:
            raise TypeError("Cannot clean up cache if no cache given.")
        if data_time is not None:
            raise NotImplementedError(f"Cannot delete data with `data_time` spec.")

        args = tuple(map(lambda x: x is None, (delta, dir_size, data_time)))
        if all(args):
            raise ValueError("One of `delta`, `dir_size` or `data_time` must be specified.")

        if isinstance(delta, (int, float)) or (isinstance(delta, str) and delta.isdigit()):
            if isinstance(delta, str):
                delta = int(delta)
            delta = TimeDelta(delta, "day")
        elif isinstance(delta, tuple):
            delta = TimeDelta(*delta)
        elif isinstance(delta, TimeDelta):
            delta = delta

        extension = getattr(self.pattern, "extension", "*")

        files = list(Path(self.cache).rglob(f"*.{extension.removeprefix('.')}"))

        if dir_size:
            try:
                directory_size = FolderSize(files=files)
            except FileNotFoundError as e:
                warnings.warn(f"Unable to calculate directory size, skipping cleanup. \n{e}", RuntimeWarning)
                return

            if directory_size < ByteSize(dir_size):
                return

            for file in directory_size.limit(dir_size, key=key):
                msg = f"Deleting '{file}' to limit directory size to {dir_size!s}."
                if verbose:
                    LOG.warn(msg)
                else:
                    LOG.debug(msg)
                delete_path(file, remove_empty_dirs=True)

        if delta is not None:
            delete_older_than(files, delta, key=key, verbose=verbose, remove_empty_dirs=True)

    @abstractmethod
    def _generate(
        self,
        *args,
        **kwargs,
    ) -> xr.Dataset:
        """
        Generate Data.
        Must be overriden by child class

        Using `self.pattern.search` the generate class can find the path to be saved at.
        If data is saved during `generate` it is used.

        """
        raise NotImplementedError(f"Parent class does not implement `_generate`. Child class must.")

    def get(self, *args, **kwargs) -> xr.Dataset:
        """
        Retrieve Data given a key

        If cache is given, automatically check to see if the file is generated,
        else, generate it and return the data

        If cache is not given, just generate and return the data

        Args:
            *args (Any):
                Arguments to generate data for
            **kwargs (Any):
                Kwargs to generate with

        Returns:
            xr.Dataset: Loaded data
        """
        Process(target=self.cleanup).run()

        if self.cache is None and self.pattern_type is None:
            return self._generate(*args, **kwargs)

        try:
            return self.load(self.search(*args, **kwargs))
        except (OSError, ValueError, PermissionError) as exception:
            # try:
            data = self._generate(*args, **kwargs)
            try:
                self._save_catalog()
                self.pattern.save(data, *args)
            except PermissionError:
                pass
            return data

    def generate(self, *args, **kwargs) -> Path | list[str | Path] | dict[str, str | Path]:
        """
        Using child classes implemented `_generate` generate data, and save
        it using the pattern.

        Returns:
            (Path | list[str | Path] | dict[str, str | Path]):
                Location of saved data
        """
        pattern = self.pattern
        if self._verbose:
            print(f"Cache is generating according to: {args}, {kwargs} at {self.cache}.")

        data = self._generate(*args, **kwargs)

        # Check to see if data has been saved
        if pattern.exists(*args):
            return pattern.search(*args)

        pattern.save(data, *args)
        self._save_catalog()

        return pattern.search(*args)

    @property
    def override(self):
        """Get a context manager within which data will be overridden in the cache."""
        return ChangeValue(self, "_override", True)

    def _save_catalog(self, catalog: CatalogEntry | None = None):
        """
        Save either the given catalog or the `pattern` within the cache directory.
        """
        if self.cache is None:
            return  # Cannot save catalog if no cache given

        if catalog is None:
            catalog = self.pattern.catalog
        if catalog is None:
            return

        catalog.save(Path(self.cache) / "catalog.cat")

    def filesystem(self, querykey: Any, *args) -> Path | list[str | Path] | dict[str, str | Path]:
        """
        Search for generated data if cache is given.
        If data does not exist yet, generate it, save it, and return the path to it

        Data is generated here if cache is given so that `.series` operations, can
        work on filesystem, and thus any dask things work well.

        Args:
            querykey (Any):
                Querykey to search for / generate data for

        Returns:
            (Path | list[str | Path] | dict[str, str | Path]):
                Filepath to discovered / generated data

        Raises:
            NotImplementedError:
                If `cache` is not set, cannot cache data.
        """

        if self.cache is None and self.pattern_type is None:
            raise NotImplementedError(
                f"CachingIndex cannot retrieve data from a filesystem without a `cache` location."
            )

        pattern = self.pattern
        Process(target=self.cleanup).run()

        try:
            if pattern.exists(querykey):
                if not self._override:
                    return pattern.search(querykey)
                else:
                    delete_path(pattern.search(querykey))
        except DataNotFoundError:
            LOG.debug(f"Failed to find data despite it looking like it existed, moving to generation.")

        return self.generate(querykey, *args)

    def __del__(self):
        Process(target=self.cleanup).run()
        try:
            self._save_catalog()
        except Exception:
            pass
        if hasattr(self, "pattern"):
            del self.pattern


class CachingIndex(BaseCacheIndex, ArchiveIndex):
    """
    Standard CachingIndex which behaves like a standard archive but with cached data
    """

    pass


class TimeCachingIndex(BaseCacheIndex, TimeIndex):
    """
    Standard CachingIndex which can handle simple time based requests
    """

    pass


class CachingForecastIndex(BaseCacheIndex, ForecastIndex):
    """
    CachingIndex which is a forecast product
    """

    pass


# class CachefromIndex(CachingIndex):
#     # TODO
#     def __init__(
#         self,
#         cache: str | Path = None,
#         pattern: str | PatternIndex = None,
#         pattern_kwargs: dict = {},
#         *,
#         transforms: Transform | TransformCollection = TransformCollection(),
#         **kwargs,
#     ):
#         """Wrap an index with a cacher"""
#         super().__init__(cache, pattern, pattern_kwargs, transforms=transforms, **kwargs)
