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


# TODO: Rewrite with better design patterns
# Too much subclassing and wacky inheritance structures
# Could look into factories or builders

"""
Indexes for pyearthtools

Implements the core data structures and functions for use in pyearthtools.

See [here][pyearthtools.data.indexes] for details.

"""

from __future__ import annotations

import warnings
import datetime
import functools
import logging
from abc import abstractmethod, ABCMeta

from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional

import xarray as xr

import pyearthtools.data
import pyearthtools.utils

from pyearthtools.data.time import pyearthtoolsDatetime, TimeDelta, TimeRange

from pyearthtools.data.transforms.transform import (
    Transform,
    TransformCollection,
    FunctionTransform,
)

from pyearthtools.data.warnings import IndexWarning
from pyearthtools.data.exceptions import DataNotFoundError
from pyearthtools.data.operations import index_routines, index_operations, forecast_op
from pyearthtools.data import operations

from pyearthtools.data.indexes.utilities.mixins import (
    CallRedirectMixin,
    CatalogMixin,
)
from pyearthtools.data.indexes.utilities import open_files, dimensions

from pyearthtools.utils.context import ChangeValue

LOG = logging.getLogger("pyearthtools.data")


class Index(CallRedirectMixin, CatalogMixin, metaclass=ABCMeta):
    """
    Base Level Index to define the structure

    To use, subclass and define the `.get` function, any calls, shall be passed through.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get(self, *args, **kwargs):
        """
        Base Level `.get` call, used to retrieve data from args
        """
        raise NotImplementedError(f"'.get' functionality not implemented by {self}")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Retrieve Data with given arguments

        Examples:
            >>> pyearthtools.data.Index()(*arguments)
            'Data from Index at found from *arguments'
        """
        return self.get(*args, **kwargs)


class FileSystemIndex(Index, metaclass=ABCMeta):
    """
    Index addon to load data from a File System

    Provides basic loading functions and allows for an index to be 'searched'.
    """

    @property
    def ROOT_DIRECTORIES(self):
        if not hasattr(pyearthtools.data.archive, "ROOT_DIRECTORIES"):
            raise KeyError("pyearthtools.data.archive has not attribute 'ROOT_DIRECTORIES'")
        return pyearthtools.data.archive.ROOT_DIRECTORIES

    def load(
        self,
        files: dict[str, str | Path] | Path | list[str | Path] | tuple[str | Path],
        **kwargs,
    ) -> Any:
        """
        Load a given list of files.

        Automatically determine method to load files for file extension

        Supported:
            - netcdf
            - pandas [csv]
            - numpy

        Args:
            files (dict[str, str | Path] | Path | list[str | Path] | tuple[str | Path]):
                Files to load
            **kwargs (Any, optional):
                Kwargs passed to underlying loading function

        Raises:
            InvalidDataError:
                If an error arose when loading file

        Returns:
            (Any):
                Loaded data
        """
        return open_files(files, **kwargs)

    def search(self, *args, **kwargs) -> Path | list[str | Path] | dict[str, str | Path]:
        """
        Find file name/path, with the underlying functionality defined by discovered location.

        All arguments passed to underlying function.

        Args:
            *args (Any, optional):
                Arguments passed to underlying search function
            *kwargs (Any, optional):
                Keyword Arguments passed to underlying search function

        Returns:
            (Path | list[str | Path] | dict[str, str | Path]):
                Path to data defined by arguments
        """
        try:
            return getattr(self, pyearthtools.utils.config.get("data.search_function"))(*args, **kwargs)
        except TypeError as e:
            raise TypeError(
                "An error arose when searching for data, likely the required arguments were not given"
            ) from e

    def exists(self, *args, **kwargs) -> bool:
        """
        Check if data exists, by checking if the file from `search()` exists

        Passes all args to `search()`

        Returns:
            (bool):
                If data exists
        """

        def check_existence(file: str | Path | dict | list):
            if isinstance(file, Path):
                return file.exists()
            elif isinstance(file, str):
                return check_existence(Path(file))
            elif isinstance(file, dict):
                return all([check_existence(val) for val in file.values()])
            elif isinstance(file, (tuple, list)):
                return all([check_existence(val) for val in file])

            raise TypeError(f"Could not check existence of type {type(file)}")

        try:
            return check_existence(self.search(*args, **kwargs))
        except DataNotFoundError:
            return False

    def get(self, *args, **kwargs):
        """
        Get data by loading it from the search

        Passes all args to `search()` and all kwargs to `load()`


        Raises:
            DataNotFoundError:
                Data could not be found

        Returns:
            (Any):
                Loaded Data
        """
        try:
            return self.load(self.search(*args), **kwargs)
        except FileNotFoundError as e:
            raise DataNotFoundError(f"Data with args: {args} could not be found.") from e

    def filesystem(self, *args) -> Path | dict[str, str]:
        """
        Find datafiles given args on local filesystem.

        Must be implemented by child class to specify data.

        Can return a dictionary[str, str], tuple, list or path representing the files to load.
        """
        raise NotImplementedError(f"File system search function not implemented for class: {self}, define in child.")


class DataIndex(Index):
    """
    Index to introduce [transforms][pyearthtools.data.transforms] to data loading

    Transforms are applied on a `retrieve` or `__call__`, but not on `get`

    """

    _pyearthtools_repr = {
        "ignore": ["transforms", "preprocess_transforms"],
        "expand_attr": ["Transforms@base_transforms", "Preprocess@preprocess_transforms"],
    }

    _skip_transforms: bool = False

    def __init__(
        self,
        transforms: Transform | TransformCollection = TransformCollection(),
        *args,
        add_default_transforms: bool = True,
        preprocess_transforms: Transform | TransformCollection | Callable | None = None,
        **kwargs,
    ):
        """
        Introduce [transforms][pyearthtools.data.transforms] to data loading

        Args:
            transforms (Transform | TransformCollection, optional):
                Base Transforms to be applied to data. Defaults to TransformCollection().
            add_default_transforms (bool, optional):
                Add Default Transformations. Defaults to True

            preprocess_transforms (Transform | TransformCollection | Callable | None):
                Transforms to apply in preprocessing for datasets. Does not work on other file formats.
                Defaults to None.
        """
        super().__init__(*args, **kwargs)
        self.base_transforms = (
            pyearthtools.data.transforms.get_default_transforms() if add_default_transforms else TransformCollection()
        )
        self.base_transforms += TransformCollection(transforms)
        self.preprocess_transforms = preprocess_transforms

    def retrieve(
        self,
        *args,
        transforms: Transform | TransformCollection = TransformCollection(),
        **kwargs,
    ) -> Any:
        """
        Retrieve data using `get`, which must be implemented, and apply both base and given transforms

        Args:
            transforms (Transform | TransformCollection, optional):
                Extra transforms to apply. Defaults to TransformCollection().

        Returns:
            (Any):
                Loaded data with transforms applied
        """
        transforms = self.base_transforms + TransformCollection(transforms)

        kwargs.update(self._get_preprocess(kwargs.pop("preprocess", None)))  # type: ignore

        if self._skip_transforms:
            return self.get(*args, **kwargs)
        return transforms(self.get(*args, **kwargs))

    def _get_preprocess(
        self, preprocess: Callable | None
    ) -> dict[Literal["preprocess"], TransformCollection | Callable]:
        """
        Get dictionary to update with preprocess kwarg set.

        If not given in init and `preprocess` return empty dict,
        otherwise merge given preprocesses

        Args:
            preprocess (Callable | None):
                Add on for preprocess functions

        Returns:
            (dict[Literal['preprocess'], TransformCollection | Callable]):
                Either empty dict or one with preprocess given
        """
        if self.preprocess_transforms is None:
            if preprocess is None:
                return {}
            return {"preprocess": preprocess}

        preprocess_transforms = TransformCollection(self.preprocess_transforms)
        if preprocess is not None:
            if isinstance(preprocess, TransformCollection):
                preprocess_transforms += preprocess
            else:
                preprocess_transforms += FunctionTransform(preprocess)
        return {"preprocess": preprocess_transforms}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.retrieve(*args, **kwargs)


class SingleTimeIndex(Index):
    """
    Introduce single time based Indexing with [pyearthtoolsDatetime][pyearthtools.data.time.pyearthtoolsDatetime].

    While [Index][pyearthtools.data.indexes.indexes.Index] assumes nothing about the selection arguments,
    this will attempt to convert them to a [pyearthtoolsDatetime][pyearthtools.data.time.pyearthtoolsDatetime], and select that time
    from the data.

    [pyearthtoolsDatetime][pyearthtools.data.time.pyearthtoolsDatetime] keeps a record of the resolution of the given date string,
    which allows for more informative warnings.

    """

    def __init__(
        self,
        data_interval: tuple[int, str] | str | int | TimeDelta | None = None,
        round: bool = False,
        **kwargs,
    ):
        """
        Setup TimeIndex,

        Will warn a user if date is of incorrect resolution

        Args:
            data_interval (tuple[int, str] | int, optional):
                Interval of data. Must follow format for
                [TimeDelta][pyearthtools.data.time.TimeDelta]. by default None
                E.g.
                    (1, 'h') = 1 Hour
                    (10, 'D') = 10 Days.
                    10 = 10 minutes.

                Defaults to None.
            round (bool, optional):
                Default value for round when retrieving data.
                Defaults to False.
        """
        super().__init__(**kwargs)

        self.set_interval(data_interval)
        self._round = round

    def _get_interval(self, interval: TimeDelta | int | str | tuple[int | float, str] | None) -> TimeDelta:
        """
        Get interval of data,

        If class has `data_interval` set use the max, or if not use provided.

        Args:
            interval (TimeDelta):
                Interval as provided to above function

        Raises:
            ValueError:
                If interval is None

        Returns:
            (TimeDelta):
                TimeDelta to use
        """

        if hasattr(self, "data_interval") and self.data_interval is not None and self.data_resolution is not None:
            if interval is None:
                interval = TimeDelta(self.data_interval)
            else:
                interval = max(TimeDelta(interval), TimeDelta(self.data_interval))

            if interval.resolution > self.data_resolution:  # Higher Resolution
                warnings.warn(
                    f"Data requested at a higher resolution than available. {interval.resolution} > {self.data_resolution}",
                    IndexWarning,
                )

        if interval is None:
            raise ValueError("If DataIndex has no 'data_interval', 'interval' must be provided")
        return TimeDelta(interval)

    def set_interval(self, data_interval: tuple[float, str] | str | int | TimeDelta | None = None):
        """
        Set interval of data

        Args:
            data_interval (tuple[int, str] | int, optional):
                Interval of data. Must follow format for
                [TimeDelta][pyearthtools.data.time.TimeDelta]. by default None
                E.g.
                    (1, 'h') = 1 Hour
                    (10, 'D') = 10 Days.
                     10 = 10 minutes.

                Defaults to None.
        """
        resolution = None
        if data_interval:
            data_interval = TimeDelta(data_interval)
            resolution = data_interval.resolution

        self.update_initialisation(data_interval=data_interval)

        self.data_interval = data_interval
        self.data_resolution = resolution

    def retrieve(
        self,
        querytime: str | pyearthtoolsDatetime,
        *args,
        select: bool = False,
        round: bool | None = None,
        **kwargs,
    ) -> Any:
        """
        Retrieve Data at given timestep, uses [Index][pyearthtools.data.indexes.Index] to load data.

        While [Index][pyearthtools.data.indexes.Index] assumes nothing, this will attempt to select time.

        Args:
            querytime (str | datetime.datetime | pyearthtoolsDatetime):
                Timestep to retrieve data at
            select (bool, optional):
                Select `querytime` in dataset. Defaults to False.
            round (bool, optional):
                Select nearest time, when selecting. Can be configured in `init`.
                Defaults to False.

        Returns:
            (Any):
                Loaded data, with time selected
        """
        querytime = pyearthtoolsDatetime(querytime)
        if self.data_resolution and querytime.resolution < self.data_resolution:
            warnings.warn(
                f"Data requested at a lower resolution than data exists at. {querytime.resolution} < {self.data_resolution}. \n"
                f"Increasing `querytime`:{querytime} resolution to {querytime.at_resolution(self.data_resolution)!r}",
                IndexWarning,
            )
            querytime = querytime.at_resolution(self.data_resolution)

        retrieval_function = getattr(super(), "retrieve", super().get)
        data = retrieval_function(querytime, *args, **kwargs)

        if not isinstance(data, (xr.Dataset, xr.DataArray)):
            return data

        time_dim = dimensions.identify_time_dimension(data)

        round = round or self._round
        if time_dim not in data.dims and time_dim in data.coords:
            data = data.expand_dims(time_dim)

        if select and time_dim in data:
            try:
                data = data.sel(
                    **{time_dim: str(pyearthtoolsDatetime(querytime))},
                    method="nearest" if round else None,
                )
            except KeyError:
                warnings.warn(
                    f"Could not find time in dataset to select on. {querytime!r}",
                    IndexWarning,
                )

        if time_dim not in data.dims and time_dim in data.coords:
            data = data.expand_dims(time_dim)

        return data


class TimeIndex(SingleTimeIndex):
    """
    Introduce general time based Indexing with [pyearthtoolsDatetime][pyearthtools.data.time.pyearthtoolsDatetime].

    Allow for multiple time retrievals.
    """

    @functools.wraps(index_routines.series)
    def series(
        self,
        start: str | pyearthtoolsDatetime,
        end: str | pyearthtoolsDatetime,
        interval: TimeDelta | tuple[int | float, str] | int | None = None,
        *,
        transforms: TransformCollection | Transform = TransformCollection(),
        **kwargs,
    ) -> xr.Dataset:
        """
        API Function of [series][pyearthtools.data.index_operations.index_routines.series] for each AdvancedTimeIndex

        Args:
            start (str | pyearthtoolsDatetime):
                Start time for series
            end (str | pyearthtoolsDatetime):
                End time for series
            interval (TimeDelta, optional):
                Interval to retrieve data at. Defaults to initialise resolution.
            transforms (TransformCollection | Transform, optional):
                Extra Transforms to apply. Defaults to TransformCollection().

        Returns:
            (xr.Dataset):
                Loaded series of data
        """

        interval = self._get_interval(interval)
        tolerance = kwargs.pop("tolerance", getattr(self, "data_interval", None))

        return index_routines.series(
            self,
            start,
            end,
            interval,
            transforms=transforms,
            tolerance=tolerance,
            **kwargs,
        )

    @functools.wraps(index_routines.safe_series)
    def safe_series(
        self,
        start: str | pyearthtoolsDatetime,
        end: str | pyearthtoolsDatetime,
        interval: TimeDelta | tuple[int | float, str] | None = None,
        *,
        transforms: TransformCollection | Transform = TransformCollection(),
        **kwargs,
    ) -> xr.Dataset:
        """
        API Function of [safe_series][pyearthtools.data.index_operations.index_routines.safe_series] for each AdvancedTimeIndex.

        Provides a safer way into get a series of data.

        Args:
            start (str | pyearthtoolsDatetime):
                Start time for series
            end (str | pyearthtoolsDatetime):
                End time for series
            interval (tuple[int, str], optional):
                Interval to retrieve data at. Defaults to initialised resolution .
            transforms (TransformCollection | Transform, optional):
                Extra Transforms to apply. Defaults to TransformCollection().

        Returns:
            (xr.Dataset):
                Loaded safe_series of data
        """
        interval = self._get_interval(interval)

        tolerance = kwargs.pop("tolerance", getattr(self, "data_interval", None))

        return index_routines.safe_series(
            self,
            start,
            end,
            interval,
            transforms=transforms,
            tolerance=tolerance,
            **kwargs,
        )

    @functools.wraps(index_operations.aggregation)
    def aggregation(
        self,
        start: str | pyearthtoolsDatetime,
        end: str | pyearthtoolsDatetime,
        interval: TimeDelta | tuple[int | float, str] | None = None,
        *,
        transforms: TransformCollection | Transform = TransformCollection(),
        **kwargs,
    ) -> xr.Dataset:
        """
        API Function of [aggregation][pyearthtools.data.index_operations.index_operations.aggregation] for each AdvancedTimeIndex

        Args:
            start (str | pyearthtoolsDatetime):
                Start time for series
            end (str | pyearthtoolsDatetime):
                End time for series
            interval (tuple[int, str], optional):
                Interval to retrieve data at. Defaults to initialise resolution .
            transforms (TransformCollection | Transform, optional):
                Extra Transforms to apply. Defaults to TransformCollection().

        Returns:
            (xr.Dataset):
                Aggregation of data
        """
        interval = self._get_interval(interval)
        if self.data_resolution:
            start = pyearthtoolsDatetime(start).at_resolution(self.data_resolution)
        # transforms = self.base_transforms + transforms

        return index_operations.aggregation(self, start, end, interval, transforms=transforms, **kwargs)

    @functools.wraps(index_operations.find_range)
    def range(
        self,
        start: str | pyearthtoolsDatetime,
        end: str | pyearthtoolsDatetime,
        interval: TimeDelta | tuple[int, str] | None = None,
        *,
        transforms: TransformCollection | Transform = TransformCollection(),
        **kwargs,
    ) -> dict:
        """
        API Function of [range][pyearthtools.data.index_operations.index_operations.find_range] for each AdvancedTimeIndex

        Args:
            start (str | pyearthtoolsDatetime):
                Start time for series
            end (str | pyearthtoolsDatetime):
                End time for series
            interval (tuple[int, str], optional):
                Interval to retrieve data at. Defaults to initialise resolution .
            transforms (TransformCollection | Transform, optional):
                Extra Transforms to apply. Defaults to TransformCollection().

        Returns:
            (dict):
                Range of data
        """
        interval = self._get_interval(interval)

        return index_operations.find_range(self, start, end, interval, transforms=transforms, **kwargs)


class SingleTimeDataIndex(TimeIndex, DataIndex):
    """
    Combine `SingleTimeIndex` and `DataIndex`,

    Allows temporal indexing with transforms applied.
    """

    def __init__(
        self,
        transforms: Transform | TransformCollection = TransformCollection(),
        preprocess_transforms: Transform | TransformCollection | Callable | None = None,
        data_interval: tuple[int, str] | int | str | TimeDelta | None = None,
        **kwargs,
    ):
        """
        Setup TimeDataIndex

        For indexing with time and applying transforms

        Args:
            transforms (Transform | TransformCollection, optional):
                Transforms to add when retrieving data. Defaults to TransformCollection().
            data_interval (tuple[int, str] | int, optional):
                Temporal Interval of Data. Defaults to None.
            preprocess_transforms (Transform | TransformCollection, optional):
                Transforms to apply in preprocessing for datasets. Does not work on other file formats.
                Defaults to None.
        """
        super().__init__(
            transforms=transforms,
            data_interval=data_interval,
            preprocess_transforms=preprocess_transforms,
            **kwargs,
        )

    def retrieve(self, *args, transforms=None, **kwargs):
        transforms = self.base_transforms + TransformCollection(transforms)
        # kwargs.update(self._get_preprocess(kwargs.pop("preprocess", None)))  # type: ignore
        with ChangeValue(self, "_skip_transforms", True):
            # Skip transforms, so that they are only applied once
            # By applying transforms after time retrieve, prevents more then necessary data going to transforms
            return transforms(super().retrieve(*args, **kwargs))


class TimeDataIndex(SingleTimeDataIndex):

    def _data_wrapper(method: Callable):  # type: ignore
        def wrapped(self, *args, **kwargs):
            if not hasattr(super(), str(method.__name__)):
                raise AttributeError(f"{self.__class__} has no attribute {method.__name__}.")

            kwargs.update(self._get_preprocess(kwargs.pop("preprocess", None)))
            kwargs["transforms"] = self.base_transforms + kwargs.pop("transforms", None)
            return getattr(super(), str(method.__name__))(*args, **kwargs)

        return wrapped

    @_data_wrapper
    def series(self, *args, **kwargs) -> xr.Dataset: ...  # type: ignore

    @_data_wrapper
    def safe_series(self, *args, **kwargs) -> xr.Dataset: ...  # type: ignore

    @_data_wrapper
    def range(self, *args, **kwargs): ...  # type: ignore

    @_data_wrapper
    def aggregation(self, *args, **kwargs) -> xr.Dataset: ...  # type: ignore


class AdvancedTimeIndex(TimeIndex):
    """
    Extend Time based indexing for Advanced uses, using the provided `data_interval`

    Overrides `retrieve`, to allow a series of data to be retrieved based upon given date resolution.

    ??? tip "New retrieve Behaviour"
        Consider a dataset with 10 minute resolution

        | Date      | Behaviour               |
        | --------- | ----------------          |
        |`2021-01-01T00:00`|Exact Data            |
        |`2021-01-01T00`   |All Data in that hour  |
        |`2021-01-01`      |All Data in that day  |
        |`2021-01`         |All Data in that month|
        |`2021`            |All Data in that year |

    !!! Important
        Many features of this class require the `data_interval` to be specified

    """

    def retrieve(
        self,
        querytime: str | datetime.datetime | pyearthtoolsDatetime,
        *,
        aggregation: str | None = None,
        select: bool = True,
        use_simple: bool = False,
        **kwargs,
    ) -> xr.Dataset:
        """
        Retrieve data at timestep, but will use the resolution of the time to infer large scale retrievals.

        !!! tip "Date Behaviour"

            | Date      | Behaviour               |
            | --------- | ----------------          |
            |`2021-01-01T00:00`|Exact Data            |
            |`2021-01-01`      |All Data in that day  |
            |`2021-01`         |All Data in that month|
            |`2021`            |All Data in that year |

        Args:
            querytime (str | datetime.datetime):
                Timestep to retrieve data at, can be exact data or range as described above.
            aggregation (str, optional):
                If data becomes a range, can specify an aggregation method. Defaults to None.
            select (bool, optional):
                Whether to attempt to select the given timestep if date is either fully qualified or data_interval not given. Defaults to True.
            use_simple (bool, optional):
                Whether to simply use the DataIndex.retrieve instead. Defaults to False.
            kwargs (Any, optional):
                Kwargs passed to downstream retrieval function

        Raises:
            DataNotFoundError:
                If Data not found at timestep

        Returns:
            (xr.Dataset):
                Loaded Dataset with transforms applied, and aggregated if `aggregation_method` given

        Note:
            Extra transforms can be supplied, using `transforms = `
        """

        querytime = pyearthtoolsDatetime(querytime)

        if not hasattr(self, "data_resolution") or not self.data_resolution or use_simple:
            data = super().retrieve(querytime, select=select, **kwargs)
            return data  # selectdata(querytime, data)

        if self.data_resolution:
            if querytime.resolution > self.data_resolution:  # Higher Resolution
                warnings.warn(
                    f"Data requested at a higher resolution than available. {querytime.resolution} > {self.data_resolution}",
                    IndexWarning,
                )
                querytime = querytime.at_resolution(self.data_resolution)
                data = super().retrieve(querytime, select=select, **kwargs)
                return data  # selectdata(querytime, data)

            elif querytime.resolution == self.data_resolution:  # Equal Resolution
                data = super().retrieve(querytime, select=select, **kwargs)
                return data

            start_time = querytime.at_resolution(self.data_resolution)
        else:
            start_time = querytime

        # Lower Resolution, find via series
        end_time = pyearthtoolsDatetime(querytime)
        end_time += 1  # Automatically adds one to the last defined date

        if self.data_resolution:
            end_time = end_time.at_resolution(self.data_resolution)

        try:
            all_data = self.series(
                start_time,
                end_time,
                interval=self.data_interval,
                skip_invalid=kwargs.pop("skip_invalid", True),
                # subset_time=kwargs.pop("subset_time", False),
                **kwargs,
            )
        except DataNotFoundError as e:
            raise DataNotFoundError(f"No Data found at {querytime}. Ensure data exists.") from e

        time_dim = dimensions.identify_time_dimension(all_data)
        # try:
        #     if time_dim in all_data:
        #         all_data = all_data.sel(time=str(querytime))
        # except KeyError:
        #     pass

        if aggregation:
            all_data = pyearthtools.data.transforms.aggregation.over(dimension=time_dim, method=aggregation)(all_data)
        return all_data

    def __call__(self, *args, **kwargs) -> Any:
        """API Function to allow easier indexing into the DataIndexes

        Options:

        | Arguments | Operation |
        | --------- | --------- |
        |`querytime` or time passed as first arg | [retrieve][pyearthtools.data.indexes.AdvancedTimeIndex] indexing |
        | [Dataset][xarray.Dataset] or [DataArray][xarray.DataArray] | Infer Spatial and Temporal Extent |
        | All else | [series][pyearthtools.data.indexes.AdvancedTimeIndex.series] indexing |

        Raises:
            KeyError:
                If using [Dataset][xarray.Dataset] method and no time dim given

        Returns:
            (Any):
                Returned Data
        """

        if "querytime" in kwargs or len(args) == 0:
            return self.retrieve(*args, **kwargs)

        if len(args) >= 1 and isinstance(args[0], (xr.Dataset, xr.DataArray)):
            ds = args[0]
            ds_time_dim = dimensions.identify_time_dimension(ds)

            if ds_time_dim not in ds.coords:
                raise KeyError("If passing dataset to get data at the same time as, it must have a 'time' coordinate.")

            transforms = kwargs.pop("transforms", TransformCollection())
            try:
                transforms += pyearthtools.data.transforms.coordinates.get_longitude(ds, transform=True)  # type: ignore
            except ValueError as e:
                LOG.debug(f"An error arose identifying the 'longitude' coordinate. {e}")
            transforms += pyearthtools.data.transforms.region.like(ds)

            time_values = ds[ds_time_dim].values

            if not isinstance(time_values, Iterable):
                time_values = pyearthtoolsDatetime(time_values)
                if self.data_resolution:
                    time_values = time_values.at_resolution(self.data_resolution)
                return self.retrieve(time_values, transforms=transforms, **kwargs)

            start_time = pyearthtoolsDatetime(time_values[0])
            end_time = pyearthtoolsDatetime(time_values[-1])

            if self.data_resolution:
                start_time = start_time.at_resolution(self.data_resolution)
                end_time = end_time.at_resolution(self.data_resolution)

            interval = (end_time - start_time) / (max(1, len(ds[ds_time_dim]) - 1))

            if start_time == end_time:
                return self.retrieve(start_time, transforms=transforms, **kwargs)

            kwargs["inclusive"] = True

            return self.series(
                start_time,
                end_time,
                interval,
                transforms=transforms,
                **kwargs,
            )

        if len(args) == 1:
            return self.retrieve(*args, **kwargs)

        return self.series(*args, **kwargs)


class AdvancedTimeDataIndex(AdvancedTimeIndex, TimeDataIndex):
    """
    Combine `AdvancedTimeIndex` and `DataIndex`,

    Allows advanced temporal indexing with transforms applied.
    """


class DataFileSystemIndex(DataIndex, FileSystemIndex):
    """
    Indexer to combine transforms and file system searching

    Combines `DataIndex` and `FileSystemIndex`, to allow transforms and
    searching on filesystems.
    """

    pass


class BaseTimeIndex(TimeIndex, DataFileSystemIndex):
    """
    Indexer to combine transforms, file system searching and basic Time

    Combines `TimeIndex`, `DataIndex` and `FileSystemIndex`, to allow transforms and
    searching on filesystems based on times.
    """

    pass


class ArchiveIndex(AdvancedTimeDataIndex, FileSystemIndex):
    """
    Default Archive Indexer, for use by on disk datasets.

    Combines `DataIndex`, `FileSystemIndex` and `AdvancedTimeIndex`, to allow transforms,
    searching, and advanced temporal indexing.

    !!! Help "Initialisation Arguments"
        transform

    """

    @functools.wraps(FileSystemIndex.search)
    def search(self, *args):
        """
        Attempt to convert first arg to a [pyearthtoolsDatetime][pyearthtools.data.time.pyearthtoolsDatetime],
        if conversion fails, ignore and continue

        Will operate with time resolution behaviour from `AdvancedTimeIndex`.
        """
        args = list(args)
        if len(args) > 0:
            try:
                date = pyearthtoolsDatetime(args[0])
                if date:
                    args[0] = date

                if not hasattr(self, "data_resolution") or not self.data_resolution:
                    return super().search(*args)

                if date.resolution > self.data_resolution:  # Higher Resolution
                    warnings.warn(
                        f"Data requested at a higher resolution than available. {date.resolution} > {self.data_resolution}",
                        IndexWarning,
                    )
                    args[0] = date.at_resolution(self.data_resolution)
                    return super().search(*args)

                elif date.resolution == self.data_resolution:  # Equal Resolution
                    return super().search(*args)

                start_date = date.at_resolution(self.data_resolution)
                func = super().search

                files = []
                results = []
                for time in TimeRange(
                    start_date,
                    (date + 1).at_resolution(self.data_resolution),
                    self.data_interval or (1, self.data_resolution),
                ):
                    search_result = func(time, *args[1:])

                    if isinstance(search_result, (str, Path)):
                        if search_result not in files:
                            results.append(search_result)
                        files.append(search_result)

                    elif isinstance(search_result, (list, tuple)):
                        for file in (file for file in search_result.values() if file not in files):
                            results.append(file)
                        files.extend(search_result)
                    elif isinstance(search_result, dict):
                        for key, val in ((key, val) for key, val in search_result.items() if val not in files):
                            results.append({key: val})
                        files.extend(list(search_result.values()))
                return results

            except Exception:
                pass
        return super().search(*args)


class ForecastIndex(TimeIndex, DataFileSystemIndex):
    """
    Index into Forecast data, where Temporal indexing and selection is invalid.

    Combines `DataIndex`, `FileSystemIndex` and `TimeIndex`.

    """

    @functools.wraps(FileSystemIndex.search)
    def search(self, *args) -> Path:
        """
        Attempt to convert first arg to a [pyearthtoolsDatetime][pyearthtools.data.time.pyearthtoolsDatetime],
        if conversion fails, ignore and continue
        """
        args = list(args)
        if len(args) > 0:
            try:
                date = pyearthtoolsDatetime(args[0])
                if date:
                    args[0] = date
            except Exception:
                pass
        return super().search(*args)

    @functools.wraps(forecast_op.forecast_series)
    def series(
        self,
        start: str | pyearthtoolsDatetime,
        end: str | pyearthtoolsDatetime,
        interval: Optional[TimeDelta | tuple[int, str]] = None,
        *args,
        **kwargs,
    ):
        interval = self._get_interval(interval)

        return forecast_op.forecast_series(self, start, end, interval, *args, **kwargs)

    def retrieve(
        self,
        basetime: str | pyearthtoolsDatetime,
        *args,
        querytime: str | pyearthtoolsDatetime | TimeDelta | None = None,
        **kwargs,
    ) -> Any:
        """
        Retrieve data from a forecast product, allowing seperate specification of basetime and querytime

        Args:
            basetime (str | pyearthtoolsDatetime):
                Basetime to get forecast from
            querytime (str | pyearthtoolsDatetime | None, optional):
                Time to select from forecast. Defaults to None.

        Raises:
            IndexError:
                If Unable to select

        Returns:
            (Any):
                Retrieved data
        """
        data = super().retrieve(basetime, *args, select=False, **kwargs)
        time_dim = dimensions.identify_time_dimension(data)

        if querytime:
            if isinstance(querytime, (tuple, TimeDelta)):
                querytime = pyearthtoolsDatetime(basetime) + TimeDelta(querytime)

            if isinstance(data, (xr.Dataset, xr.DataArray)) and time_dim in data:
                data = data.sel(**{time_dim: [str(querytime)]})
            else:
                raise IndexError(f"Unable to select `time` on {data}")
        return data

    def aggregation(
        self,
        querytime: str | pyearthtoolsDatetime,
        aggregation: str | Callable,
        *,
        preserve_dims: list | None = None,
        reduce_dims: list | None = None,
        transforms: TransformCollection | Transform = TransformCollection(),
        **kwargs,
    ) -> xr.Dataset:
        """
        API Function of [aggregation][pyearthtools.data.index_operations.index_operations.aggregation] for each ForecastIndex

        Args:
            querytime (str | pyearthtoolsDatetime):
                Time to get data at
            aggregation (str | Callable):
                Aggregation method to apply.
            transforms (TransformCollection | Transform, optional):
                Extra Transforms to apply. Defaults to TransformCollection().

        Returns:
            (xr.Dataset):
                Aggregation of data
        """

        return operations.aggregation(
            transforms(self(querytime)),
            aggregation=aggregation,
            preserve_dims=preserve_dims,
            reduce_dims=reduce_dims,
            **kwargs,
        )


class StaticDataIndex(DataFileSystemIndex):
    """
    Index into Static Data
    """

    pass
