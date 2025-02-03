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

import sys
from abc import abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Callable
import warnings
import logging
from hashlib import sha512
import time
from multiprocessing import Process

import xarray as xr

import pyearthtools.data

from pyearthtools.data import patterns, TimeDelta, DataNotFoundError
from pyearthtools.data.transforms import Transform, TransformCollection
from pyearthtools.data.patterns.default import PatternIndex
from pyearthtools.data.warnings import pyearthtoolsDataWarning

from pyearthtools.data.indexes import (
    ArchiveIndex,
    FileSystemIndex,
    ForecastIndex,
    TimeIndex,
    DataIndex,
    Index,
)
from pyearthtools.data.indexes.utilities.delete_files import delete_older_than, delete_path
from pyearthtools.data.indexes.utilities.folder_size import ByteSize, FolderSize

from pyearthtools.utils.context import ChangeValue

LOG = logging.getLogger("pyearthtools.data")
OVERRIDE = False


class BaseCacheIndex(DataIndex):
    """
    Base CacheIndex

    Cannot be used directly, see `MemCache` or `FileSystemCacheIndex`.
    """

    _override: bool = False

    @property
    def override(self):
        """Get a context manager within which data will be overridden in the cache."""
        return ChangeValue(self, "_override", True)

    @property
    def global_override(self):
        """Get a context manager within which data will be overridden in all caches."""
        from pyearthtools.data.indexes import cacheIndex

        return ChangeValue(cacheIndex, "OVERRIDE", True)

    @abstractmethod
    def _generate(
        self,
        *args,
        **kwargs,
    ) -> xr.Dataset:
        """
        Generate Data.
        Must be overriden by child class
        """
        raise NotImplementedError("Parent class does not implement `_generate`. Child class must.")


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, (xr.DataArray, xr.Dataset)):
        size += obj.nbytes
    elif isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


class MemCache(BaseCacheIndex):
    """
    Memory Cache

    ## Example
    ```python
    import pyearthtools.data

    mem_cache = pyearthtools.data.indexes.FunctionalMemCacheIndex(function = pyearthtools.data.archive.ERA5.sample())
    mem_cache_test('2000-01-01T00')
    # Cached into memory
    ```
    """

    _cache: dict[str, Any]
    _access_time: dict[str, float]

    def __init__(
        self,
        pattern: str | type | PatternIndex | None = None,
        pattern_kwargs: dict[str, Any] | None = None,
        *,
        max_size: str | ByteSize | None = None,
        compute: bool = False,
        transforms: Transform | TransformCollection = TransformCollection(),
        add_default_transforms: bool = True,
        **kwargs,
    ):
        """
        Cache into memory

        Uses either hash of args and kwargs or `pattern` to create key,

        Args:
            pattern (str | type | PatternIndex | None, optional):
                Pattern to use to create path to act as key. Defaults to None.
            pattern_kwargs (dict[str, Any] | None, optional):
                Kwargs for `pattern` if given. Defaults to None.
            max_size (str | ByteSize | None, optional):
                Max size of cache, set to None for no limit. Defaults to None.
            compute (bool, optional):
                Compute xarra / dask objects when given. Defaults to False.
            transforms (Transform | TransformCollection, optional):
                Transforms to add upon data retrieval. Defaults to TransformCollection().
        """

        self._pattern = pattern
        self.pattern_kwargs = pattern_kwargs
        self._max_size = max_size if max_size is None else ByteSize(max_size)
        self._compute = compute

        super().__init__(transforms=transforms, add_default_transforms=add_default_transforms, **kwargs)
        self.record_initialisation()

        self._cache = {}
        self._access_time = {}

    @property
    def size(self):
        """Size of current cache,

        Will fully count size of `xarray` objects even if delayed
        """
        return get_size(self._cache)

    def cleanup(self, complete: bool = False):
        """
        Cleanup cache, limiting size to `max_size` if given.

        Args:
            complete (bool, optional):
                Completely remove cache. Defaults to False.
        """
        if complete:
            self._cache = {}

        if self._max_size is None:
            return

        while self.size > self._max_size:
            sorted_access_times = sorted(self._access_time.items(), key=lambda x: x[1] - time.time(), reverse=False)
            if len(sorted_access_times) <= 1:
                break  # Break out if only 1 entry
            oldest_key = sorted_access_times[0][0]

            self._access_time.pop(oldest_key, None)
            self._cache.pop(oldest_key, None)

    @cached_property
    def pattern(self) -> PatternIndex | None:
        """Get Pattern from `__init__` args"""

        if self._pattern is None:
            return None

        pattern_kwargs = dict(self.pattern_kwargs or {})
        pattern_kwargs["root_dir"] = pattern_kwargs.pop("root_dir", "/")

        if isinstance(self._pattern, str):
            return getattr(patterns, self._pattern)(**pattern_kwargs)
        elif isinstance(self._pattern, PatternIndex):
            return self._pattern
        elif isinstance(self._pattern, type):
            return self._pattern(**pattern_kwargs)
        else:
            raise TypeError(f"Cannot parse `pattern` of {type(self._pattern)}")

    def get_hash(self, *args) -> str:
        """
        Get hash of args for unique key of data

        If `pattern` is set, use it to create a path.
        """
        if self.pattern is not None:
            pattern_path = self.pattern.search(*args)
            return str(pattern_path)
        return sha512(bytes(str("-".join(str(a) for a in args)), "utf-8")).hexdigest()

    def get(self, *args, **kwargs):
        """
        Get data from Memory Cache
        """
        hash_value = self.get_hash(*args)

        self._access_time[hash_value] = time.time()
        self.cleanup()

        if hash_value in self._cache and (not self._override or not OVERRIDE):
            return self._cache[hash_value]

        data = self._generate(*args, **kwargs)
        if hasattr(data, "compute") and self._compute:
            data = data.compute()

        self._cache[hash_value] = data
        return self._cache[hash_value]


class FileSystemCacheIndex(BaseCacheIndex, FileSystemIndex):
    """
    DataIndex Object that has no data on disk initially,
    but is being generated from other sources and saved in given cache.


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

    _cleanup: dict[str, Any] | float | int | str | None = None
    _save_self = True  # Save self as `index.cat` when saving

    def __init__(
        self,
        cache: str | Path | None,
        pattern: str | type | PatternIndex | None = None,
        pattern_kwargs: dict[str, Any] | str = {},
        *,
        transforms: Transform | TransformCollection = TransformCollection(),
        cleanup: dict[str, Any] | float | int | str | None = None,
        override: bool | None = None,
        verbose: bool = False,
        save_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        Base FileSystemCacheIndex Object to Cache data on the fly

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
                String of pattern to use or defined pattern.
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
            override (bool, optional):
                Override cached data. Defaults to False.
            save_kwargs (dict[str, Any], optional):
                Kwargs to pass to save function. Defaults to None.

        Raises:
            ValueError: If `cache` and `pattern` not given.
        """
        base_transform = TransformCollection() + transforms
        if pattern_kwargs and "extension" in pattern_kwargs:
            kwargs["add_default_transforms"] = kwargs.pop("add_default_transforms", "nc" in pattern_kwargs["extension"])  # type: ignore
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
        self._save_kwargs = save_kwargs or {}

        if "data_interval" in kwargs:
            self.pattern_kwargs["data_interval"] = kwargs["data_interval"]

        if override is not None:
            warnings.warn(
                "Override is deprecated, use `.override` for context manager",
                DeprecationWarning,
            )

        if self._input_cache is None and self.pattern_type is None:
            warnings.warn(
                "Without a `cache` nor `pattern` given, this `CachingIndex` will not cache.",
                UserWarning,
            )

        _ = self.pattern

        # self._save_catalog(self.catalog, 'index')

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
            raise AttributeError("`cache` nor `pattern` were provided on `init`, so no `Pattern` can be found.")

        pattern_kwargs = dict(self.pattern_kwargs)

        def _update_kwargs(spec_pattern: type, **kwargs: Any) -> dict[str, Any]:
            search_location, _ = patterns.utils.parse_root_dir(self._input_cache)  # type: ignore
            if search_location is None or not Path(search_location).exists():
                return kwargs
            try:
                loaded_catalog = pyearthtools.data.load(search_location)
            except FileNotFoundError:
                return kwargs
            except Exception as e:
                warnings.warn(f"An error occurred updating kwargs from existing cache,\n{e}", pyearthtoolsDataWarning)
                return kwargs

            if not isinstance(loaded_catalog, PatternIndex):
                return kwargs

            if type(loaded_catalog) == spec_pattern:
                _kwargs = loaded_catalog.initialisation
                _kwargs.update(**kwargs)
                _kwargs.pop("__args", None)
                _kwargs.pop("root_dir")
                return _kwargs
            return kwargs

        if self._input_cache is not None and self.pattern_type is None:
            if "data_interval" in pattern_kwargs:
                pattern_index = patterns.TemporalExpandedDate
                pattern_kwargs["file_resolution"] = pattern_kwargs.pop(
                    "file_resolution",
                    TimeDelta(pattern_kwargs["data_interval"]).resolution,
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
                raise ValueError("With 'pattern' as a str, and no 'cache'. Location of data is unclear.")

            pattern_kwargs = _update_kwargs(getattr(patterns, self.pattern_type), **pattern_kwargs)
            return getattr(patterns, self.pattern_type)(root_dir=self._input_cache, **pattern_kwargs)

        elif isinstance(self.pattern_type, PatternIndex):
            self._input_cache = self.pattern_type.root_dir
            return self.pattern_type

        elif isinstance(self.pattern_type, type):
            pattern_kwargs = _update_kwargs(self.pattern_type, **pattern_kwargs)

            return self.pattern_type(self._input_cache, **pattern_kwargs)

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
            if isinstance(self._cleanup, str) and "," in self._cleanup and len(self._cleanup.split(",")) == 2:
                split = self._cleanup.split(",")
                self.__run_cleanup(**{split[0]: split[1]})  # type: ignore
            else:
                try:
                    self.__run_cleanup(delta=self._cleanup)
                except TypeError:
                    self.__run_cleanup(dir_size=self._cleanup)

    def __run_cleanup(
        self,
        delta: TimeDelta | str | int | float | tuple | None = None,
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
            raise NotImplementedError("Cannot delete data with `data_time` spec.")

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
                warnings.warn(
                    f"Unable to calculate directory size, skipping cleanup. \n{e}",
                    RuntimeWarning,
                )
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
        self.save_record()

        if self.cache is None and self.pattern_type is None:
            return self._generate(*args, **kwargs)

        return self.generate(*args, **kwargs)
        try:
            return self.generate(*args, **kwargs)
            # return self.pattern.retrieve(*args, **kwargs)
        except (OSError, ValueError, PermissionError) as exception:
            raise
            LOG.warn(f"An exception occurred loading the data, {exception}.")
            data = self._generate(*args, **kwargs)
            try:
                self.pattern.save(data, *args, save_kwargs=self._save_kwargs)
            except PermissionError:
                pass
            return data

    def _check_if_exists(self, *args) -> bool:
        """Check if data exists, overriding it if it does and `OVERRIDE = True`"""
        pattern = self.pattern

        # Check to see if data has already been generated and saved
        if pattern.exists(*args):
            if self._override or OVERRIDE:
                LOG.info(f"At cache {self.cache} data was found but being overwritten.")
                delete_path(pattern.search(*args))  # type: ignore
            else:
                return True
        return False

    def generate(self, *args, **kwargs):
        """
        Using child classes implemented `_generate`, generate data, and save
        it using the pattern.

        Return the saved data as managed by the pattern.

        Only args is passed to save pattern to find the path to save at.

        Returns:
            (Any):
                Saved and reloaded data
        """
        pattern = self.pattern

        # Check to see if data has already been generated and saved
        if self._check_if_exists(*args):
            return pattern(*args)

        LOG.debug(f"Cache is generating according to: {args}, {kwargs} at {self.cache}.")

        data = self._generate(*args, **kwargs)
        pattern.save(data, *args, save_kwargs=self._save_kwargs)
        # try:
        #     pattern.save(data, *args, save_kwargs=self._save_kwargs)
        # except Exception as e:
        #     LOG.critical(f"An exception occured saving the generated data, DID NOT SAVE. {e}")
        #     return data

        return pattern(*args)

    def save_record(self):
        """
        Save record of the cache and pattern within the cache directory.
        """
        if self.cache is None:
            return  # Cannot save catalog if no cache given

        if not Path(self.cache).exists():
            Path(self.cache).mkdir(parents=True, exist_ok=True)

        self.pattern.save_index(Path(self.cache) / "catalog.cat")
        if self._save_self:
            self.save_index(Path(self.cache) / "index.cat")

    def filesystem(self, *args) -> Path | list[str | Path] | dict[str, str | Path]:
        """
        Search for generated data if cache is given.
        If data does not exist yet, generate it, save it, and return the path to it

        Data is generated here if cache is given so that `.series` operations, can
        work on filesystem, and thus any dask things work well.

        Args:
            args (Any):
                Args to search for / generate data for

        Returns:
            (Path | list[str | Path] | dict[str, str | Path]):
                Filepath to discovered / generated data

        Raises:
            NotImplementedError:
                If `cache` is not set, cannot cache data.
        """
        self.save_record()

        if self.cache is None and self.pattern_type is None:
            raise NotImplementedError("CachingIndex cannot retrieve data from a filesystem without a `cache` location.")

        pattern = self.pattern
        Process(target=self.cleanup).run()

        try:
            if self._check_if_exists(*args):
                return pattern.search(*args)
        except DataNotFoundError:
            LOG.debug("Failed to find data despite it looking like it existed, moving to generation.")

        self.generate(*args)
        return pattern.search(*args)

    def __del__(self):
        Process(target=self.cleanup).run()
        try:
            self.save_record()
            del self.pattern
        except Exception:
            pass


def CacheFactory(basecache: type, index: type[Index], *, name: str | None = None, doc: str | None = None) -> type:
    """Create Cache Subclasses"""

    class SubCache(basecache, index):
        pass

    if name:
        SubCache.__name__ = name
        SubCache.__qualname__ = name
    if doc:
        SubCache.__doc__ = doc
    return SubCache


CachingIndex: type[ArchiveIndex] = CacheFactory(
    FileSystemCacheIndex,
    ArchiveIndex,
    name="CachingIndex",
    doc="Standard CachingIndex which behaves like a standard archive but with cached data",
)

TimeCachingIndex: type[TimeIndex] = CacheFactory(
    FileSystemCacheIndex,
    TimeIndex,
    name="TimeCachingIndex",
    doc="Standard CachingIndex which can handle simple time based requests",
)

CachingForecastIndex: type[ForecastIndex] = CacheFactory(
    FileSystemCacheIndex, ForecastIndex, name="TimeCachingIndex", doc="CachingIndex which is a forecast product"
)


class FunctionalCache(BaseCacheIndex):
    # @wraps(BaseCacheIndex.__init__)
    _save_self = False

    def __init__(self, *args, function: Callable[[Any], Any], **kwargs):
        super().__init__(*args, **kwargs)
        self.record_initialisation()
        self._function = function

    def _generate(self, *args, **kwargs) -> xr.Dataset:
        return self._function(*args, **kwargs)


FunctionalCacheIndex: type[FileSystemCacheIndex] = CacheFactory(
    FunctionalCache, FileSystemCacheIndex, name="FunctionalCacheIndex"
)
FunctionalMemCacheIndex: type[MemCache] = CacheFactory(FunctionalCache, MemCache, name="FunctionalMemCacheIndex")
