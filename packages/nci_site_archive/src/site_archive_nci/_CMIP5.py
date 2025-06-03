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

"""
CMIP5 Accessor
"""

import datetime
import os
from pathlib import Path
from typing import Any, Literal
import warnings
from collections import namedtuple
import pandas as pd
import cftime

import functools

import xarray as xr

import pyearthtools.data
from pyearthtools.data.exceptions import DataNotFoundError
from pyearthtools.data.warnings import IndexWarning
from pyearthtools.data.indexes import ArchiveIndex, ForecastIndex, DataFileSystemIndex, decorators
from pyearthtools.data.time import Petdt, TimeDelta
from pyearthtools.data.transforms import Transform, TransformCollection

from pyearthtools.data.archive import register_archive

from site_archive_nci.utilities import check_project, cached_exists, cached_iterdir

from pyearthtools.data.time import Petdt, TimeDelta, time_delta_resolution, TimeRange
from pyearthtools.data.operations.utils import identify_time_dimension
from pyearthtools.data.operations.index_routines import _mf_series

# Path schema for CMIP5
NamedPath = namedtuple(
    "NamedPath", ["institute", "model", "scenario", "interval", "realm", "cat", "expid", "expver", "variable"]
)


class CMIP_Path:
    """
    Interpret the path elements of a file containing CMIP data according to the path schema
    """

    def __init__(self, *, elements=None, filepath=None):

        if not elements:
            dirpath, filename = os.path.split(filepath)
            elements = dirpath.split("/")[-9:]

        self.namedpath = NamedPath(*elements)

        if filename:
            self.filename = filename

    def __getitem__(self, keyword):
        return self.namedpath.__getattribute__(keyword)


INSTITUTIONS = [
    "BCC",
    "BNU",
    "CCCma",
    "CMCC",
    "CNRM-CERFACS",
    "FIO",
    "ICHEC",
    "INM",
    "INPE",
    "IPSL",
    "LASG-CESS",
    "LASG-IAP",
    "MIROC",
    "MOHC",
    "MPI-M",
    "MRI",
    "NASA-GISS",
    "NASA-GMAO",
    "NCAR",
    "NIMR-KMA",
    "NOAA-GFDL",
    "NOAA-NCEP",
    "NSF-DOE-NCAR",
]

UNDER_DEV_MSG = """
Under development. This class currently only allows single values - e.g. one institution, one model, one experiment id etc. Expansion to allowing multiple
values in under way, which will create a more complex DataSet object containing all relevant data in a single in-memory object.
"""


def rounder(time: Petdt, interval: int) -> str:
    hour = time.hour
    return "%02d00" % ((hour // interval) * interval,)


SAMPLE_PLUGIN_DICT = {"models": "bcc-csm1-1", "scenarios": "an", "institutions": "BCC"}


def to_cftime(times):
    """
    Given an iterable of Petdts, return an iterable of cftime.DatetimeNoLeap
    """

    pdt = times[0]

    # converted = [cftime.num2date(pdt.datetime.timestamp(), 'seconds since 1970-01-01') for pdt in times]

    converted = [pdt.to_cftime(calendar="noleap") for pdt in times]

    return converted


@register_archive("CMIP5", sample_kwargs=SAMPLE_PLUGIN_DICT)
class CMIP5(ArchiveIndex):
    """Index into Australian Community Climate and Earth-System Simulator"""

    @property
    def _desc_(self):
        return {
            "singleline": "CMIP5 Data Accessor",
            "Documentation": "https://opus.nci.org.au/spaces/CMIP/pages/26287764/CMIP+Community+Home",
        }

    def __init__(
        self,
        variables: list[str] | str,
        institutions: list[str] | str,
        scenarios: list[str] | str,
        models: list[str] | str,
        interval: list[str] | str,
        **kwargs,
    ):
        """
        Setup CMIP Accessor

        Args:
            variables: Variables to retrieve
            institution: ACCESS Region Code - ['g','bn','ad','sy','vt','ph','nq','dn']
            transforms: Base Transforms to apply. Defaults to TransformCollection().
        """
        check_project(project_code="al33")
        self.variables = [variables] if isinstance(variables, str) else variables
        self.institutions = [institutions] if isinstance(institutions, str) else institutions
        self.interval = [interval] if isinstance(interval, str) else interval
        self.models = [models] if isinstance(models, str) else models
        self.scenarios = [scenarios] if isinstance(scenarios, str) else scenarios

        self.walk_cache = []  # Caches filesystem walks for efficiency

        warnings.warn(UNDER_DEV_MSG)

        super().__init__(
            transforms=TransformCollection(),
        )

        self.record_initialisation()

    def quick_walk(self):
        """
        Walking a large filesystem to find matching filenames can take a long time.
        This function uses the query dictionary to more effectively walk only
        the parts of the filesystem actually relevant to the dataset. If performance
        is not a concern or if the filesystem is small, just use os.walk.
        """

        CMIP5_HOME = self.ROOT_DIRECTORIES["CMIP5"]
        basepath = Path(CMIP5_HOME)

        # Only walk the filesystem once, use the cache thereafter
        if self.walk_cache:
            for root, dirs, files in self.walk_cache:
                yield (root, dirs, files)

            return

        # Only walk the filesystem for configured institutions
        for institution in self.institutions:
            institution_basepath = os.path.join(basepath, institution)

            for model in self.models:
                model_basepath = os.path.join(institution_basepath, model)

                for scenario in self.scenarios:
                    scenario_basepath = os.path.join(model_basepath, scenario)

                    all_entries = list(os.walk(scenario_basepath))
                    walk_cache = []

                    for root, dirs, files in all_entries:
                        # We don't care about directories with no files
                        if files:
                            walk_cache.append((root, dirs, files))
                            yield (root, dirs, files)

                    self.walk_cache = walk_cache

    def filesystem(self, query_dictionary={}):
        """
        Given the supplied query, return all filenames which contain the data necessary to extract the data
        for the query. For example, figure out what time index and variables the user wants, and go find
        all the files matching those time indexes and variables so they can get loaded into memory and
        then the relevant data extracted and re-aggregated as needed by other parts of the system.
        """
        paths = {}
        paths_to_open = []

        # Walk the filesystem finding relevant file paths
        # A more efficient walk ordered by time or primary dimension may be more efficient
        for root, _dirs, files in self.quick_walk():

            paths = [os.path.join(root, file) for file in files]
            relevant = [p for p in paths if self.match_path(p, query_dictionary)]
            paths_to_open += relevant

        return paths_to_open

    def match_path(self, path, query):
        """
        Given a query (typically a date/time) and a full path to a filename, go and check
        if that filename is likely to contain data relevant to the query

        Returns:
            True if the path and query match
            False if the file should be ignored
        """

        path = CMIP_Path(filepath=path)

        match = True

        if path["variable"] not in self.variables:
            match = False

        if path["model"] not in self.models:
            match = False

        if path["interval"] not in self.interval:
            match = False

        if path["scenario"] not in self.scenarios:
            match = False

        if "atmos" != path["realm"]:
            match = False

        # TODO: Filter by matching time range of filename or data

        return match

    # Override the series method because of the unusual datetime class
    # Maybe the data loader should convert from unusual calendar datetimes to regular calendar datetimes
    def series(
        DataFunction: "AdvancedTimeIndex",
        start: str | Petdt,
        end: str | Petdt,
        interval: tuple[float, str] | TimeDelta,
        *,
        inclusive: bool = False,
        skip_invalid: bool = False,
        transforms: Transform | TransformCollection = TransformCollection(),
        verbose: bool = False,
        force_get: bool = False,
        subset_time: bool = True,
        time_dim: str | None = None,
        tolerance: tuple | pd.Timedelta | None = None,
        **kwargs,
    ) -> xr.Dataset:
        """
        Index into Provided Data function to create a continuous series of Data

        Args:
            DataFunction (AdvancedTimeIndex):
                Data function, must be AdvancedTimeIndex or child
            start (str | datetime.datetime | Petdt):
                Timestep to begin series at
            end (str | datetime.datetime | Petdt):
                Timestep to end series at
            interval (tuple[float, str]):
                Time interval between samples. Use pandas.to_timedelta notation, (10, 'minute')
            inclusive (bool, optional):
                Whether end time is included in retrieval. Defaults to False.
            skip_invalid (bool, optional):
                Whether to skip invalid data. Defaults to False.
            transforms (Transform | TransformCollection, optional):
                Extra [Transform's][pyearthtools.data.transforms.Transform] to be applied to data. Defaults to TransformCollection().
            verbose (bool, optional):
                Print logging messages. Defaults to False.
            force_get (bool, optional):
                Use series method which loads each dataset using `.get`.
                WARNING: Takes significantly longer, as it does not use dask. Defaults to False.
            subset_time (bool, optional):
                Whether to force subset time dim. Defaults to True.
            tolerance (tuple | pd.Timedelta, optional):
                Tolerance for time subsetting. Defaults to None.

        Returns:
            (xr.Dataset): Loaded xarray dataset
        """

        transforms = TransformCollection(transforms)

        use_single = force_get
        if not hasattr(DataFunction, "search"):
            use_single = True

        # if isinstance(DataFunction, pyearthtools.data.CachingIndex):
        #     use_single = True

        interval = TimeDelta(interval)
        start = Petdt(start)
        end = Petdt(end)

        start = start.at_resolution(max(interval.resolution, start.resolution))
        end = end.at_resolution(max(end.resolution, start.resolution))

        if DataFunction.data_resolution:
            start = start.at_resolution(min(DataFunction.data_resolution, start.resolution))
            end = end.at_resolution(min(DataFunction.data_resolution, start.resolution))

        if inclusive:
            end = end + interval

        function = _mf_series

        if use_single:
            function = _get_series
        # else:
        # transforms += getattr(DataFunction, 'base_transforms', None)

        try:
            data = function(
                DataFunction,
                start,
                end,
                interval,
                skip_invalid=skip_invalid,
                transforms=None,
                verbose=verbose,
                **kwargs,
            )
        except NotImplementedError as e:
            data = _get_series(
                DataFunction,
                start,
                end,
                interval,
                skip_invalid=skip_invalid,
                transforms=None,
                verbose=verbose,
                **kwargs,
            )

        time_dim = time_dim or identify_time_dimension(data)
        if time_dim in data:
            data = data.sortby(time_dim)

        if not isinstance(data, xr.Dataset):
            return data

        """
        Subsetting on time
        TODO: Improvements
        """

        if subset_time:
            # If resolution of data is greater than the start resolution
            # expand time to include all in 1 step of start and its resolution
            # e.g. hourly data with a daily start and a monthly interval goes from
            # start = 'year-01-01', end = 'year+1', and interval is (1, 'month')
            # ['year-01-01', 'year-02-01'] to
            # ['year-01-01T00', 'year-01-01T01', 'year-01-01T02',..., 'year-02-01T22', 'year-02-01T23']
            timesteps = list(TimeRange(start, end, interval))
            if (
                pyearthtools.utils.config.get("data.experimental")
                and DataFunction.data_resolution
                and DataFunction.data_resolution > start.resolution
            ):
                timesteps = [
                    t
                    for time in timesteps
                    for t in pyearthtools.data.TimeRange(time, time + 1, DataFunction.data_interval)
                ]

            time = list(set(map(lambda x: x.datetime64("ns"), timesteps)) & set(data[time_dim].values))

            if not time:
                time = timesteps
            sel_kwargs = {}

            if tolerance:  # and start.resolution < interval.resolution:
                if isinstance(tolerance, tuple):
                    tolerance = TimeDelta(tolerance)

                if isinstance(tolerance, TimeDelta):
                    tolerance = tolerance._timedelta

                sel_kwargs = getattr(DataFunction, "sel_kwargs", dict(method="bfill", tolerance=tolerance))
                time = list(
                    set(
                        map(
                            lambda x: x.datetime64("ns") if x < end else None,
                            timesteps,
                        )
                    )
                )
                while None in time:
                    time.remove(None)

            time.sort()

            if len(time) == 0 and verbose:
                warnings.warn(
                    f"Set of valid time is of length 0. Consider validity and resolution. For request: {start} -> {end} @ {interval}",
                    IndexWarning,
                )

            subset_ds = data

            time_orig = time
            time = to_cftime(time)

            # Try selecting exact time indexes
            # FIXME: Something in here causes a system exit without actually entering the exception block
            # try:
            #     subset_ds = data.sel(**{time_dim: time}, **sel_kwargs)
            # except Exception as ke:
            #     warnings.warn("Could not find expected datetime indexes in the data, falling back to slice operation")

            # We may be entering a non-gregorian calendar zone supported by cftime only
            calendar = subset_ds.time[0].item().calendar
            start = start.to_cftime(calendar=calendar)
            end = end.to_cftime(calendar=calendar)
            subset_ds = subset_ds.sel(**{time_dim: slice(start, end)})

            if not len(subset_ds[time_dim]) == 0:
                data = subset_ds
            else:
                warnings.warn(
                    f"When subsetting no time dimension remained, therefore, skipping the subsetting. For request: {start} -> {end} @ {interval}",
                    IndexWarning,
                )

        return transforms(data)
