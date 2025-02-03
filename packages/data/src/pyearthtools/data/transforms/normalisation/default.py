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

from abc import abstractmethod
from pathlib import Path
from typing import Callable, Hashable
import warnings
import tempfile
import logging

import xarray as xr

import pyearthtools.data

from pyearthtools.data import pyearthtoolsDatetime
from pyearthtools.data.transforms.normalisation._utils import format_class_name
from pyearthtools.data.transforms.transform import (
    FunctionTransform,
    Transform,
    get_default_transforms,
)
from pyearthtools.data.indexes.utilities.fileload import open_files

from pyearthtools.utils.initialisation.imports import dynamic_import

LOG = logging.getLogger("pyearthtools.data")


def open_file(file: str | tuple | dict):
    data = open_files(file)
    if isinstance(data, (xr.Dataset, xr.DataArray)):
        data = get_default_transforms()(data)
        data = pyearthtools.data.transforms.coordinates.drop("time", ignore_missing=True)(data)
    return data


def get_and_print(lambda_func: Callable, print_message: str, print_control: bool = True):
    """
    Get result of a function, but print a message before
    """

    def under_func(*args, **kwargs):
        if print_control:
            print(print_message)
        return lambda_func(*args, **kwargs)

    return under_func


class normaliser:
    def __init__(
        self,
        index: pyearthtools.data.AdvancedTimeIndex,
        start: pyearthtoolsDatetime | pyearthtoolsDatetime | None = None,
        end: pyearthtoolsDatetime | pyearthtoolsDatetime | None = None,
        interval: int | tuple | None = None,
        *,
        override: str | Path | dict | None = None,
        cache: str | Path | None = None,
        function: Callable | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Base Normalise Class

        Setup Transformer Class to normalise and unnormalise data

        Can't be used directly, see `Normalise`, & `Unnormalise`.

        Anomaly, Range, Deviation all require `start`, `end`, and `interval` to be given or `file`.

        Function, Log, and Manual_Range don't require any of the above listed.


        Args:
            index (pyearthtools.data.AdvancedTimeIndex, optional):
                AdvancedTimeIndex being normalised, used to get aggregation & range
            start (pyearthtoolsDatetime | pyearthtoolsDatetime, optional):
                Start Date for retrieval
            end (pyearthtoolsDatetime | pyearthtoolsDatetime, optional):
                End Date for retrieval
            interval (int | tuple, optional):
                Interval between samples. Use pandas.to_timedelta notation, (10, 'minute')
            override (str | Path | dict, optional):
                Instead of finding the appropriate data, use this instead.
                Config:
                    | Method | Type |
                    | -------- | ------ |
                    | anomaly  | netcdf file containing average of variables |
                    | temporal_difference  | netcdf file containing diff of variables |
                    | range    | json file or dictionary with variables, and/or min / max |
                    | deviation| Dictionary with two files, one 'average' and the other 'deviation ' |
                    Defaults to None.
            cache (str, optional):
                Where to save data generated. If not given, no data cached. Defaults to None.
            function (Callable, optional):
                Function to use with 'functional' normalisation. Defaults to None.
            verbose (bool, optional):
                Show status. Defaults to False.
            **kwargs (Any):
                All passed to climatology/range, whichever is used. NOTE: Thus transforms can be used
        """
        warnings.warn(
            "The transform method of normalisation will be being deprecated.",
            DeprecationWarning,
        )
        kwargs.update(skip_invalid=True)
        self.retrieval_arguments = dict(start=start, end=end, interval=interval, verbose=verbose, **kwargs)

        if cache == "temp":
            temp_dir = tempfile.TemporaryDirectory()
            cache = temp_dir.name
            self.temp_dir = temp_dir

        self.cache_dir = cache
        self.index = index

        self._function = function
        self.override = override
        self.verbose = verbose

    @property
    def _info_(self):
        return dict(
            index=self.index,
            cache=self.cache_dir,
            override=self.override,
            function=self._function,
            **self.retrieval_arguments,
        )

    def check_init_args(self):
        """
        Check that the init args given are valid.

        If not raise an error.
        """
        if self.override is not None:
            return True

        for arg in ["start", "end", "interval"]:
            if not self.retrieval_arguments[arg]:
                raise RuntimeError(
                    f"`override`, or (`start`, `end` and `interval`) was not given in `__init__`."
                    "These must be given in order to find the normalisation values."
                )
        return True

    def get_aggregation(
        self,
        variable_name: Hashable,
        method: str,
        dims: str | None = "time",
        save_prefix: str | None = None,
        **kwargs,
    ) -> xr.Dataset:
        """
        Get result of aggregation function on data using [aggregation][pyearthtools.data.AdvancedTimeIndex.aggregation]

        Args:
            variable_name (str):
                Variable name to look for
            method (str):
                Aggregation Method to use
            dims (str | None, optional):
                Dims to calculate over. Defaults to 'time'
            save_prefix (str | None, optional):
                Name to add to save file. Defaults to None

        Returns:
            (xr.Dataset):
                Standard Deviation of variable
        """

        retrieval_args = self.retrieval_arguments

        transforms = retrieval_args.pop("transform", None)  # getattr(
        #     self.DataIndex, "base_transform", TransformCollection()
        # ) + retrieval_args.pop("transform", None)

        save_pattern = (
            pyearthtools.data.patterns.ArgumentExpansion(
                Path(self.cache_dir) / f"{str(save_prefix)+'_' if save_prefix else ''}{method}",
                extension=".nc",
            )
            if self.cache_dir
            else None
        )

        # aggregated_data = get_and_print(
        #     lambda: self.index.aggregation(
        #         **retrieval_args,
        #         transforms=transforms,
        #         aggregation=method,
        #         aggregation_dim=dims,
        #     ),
        #     f"Getting {method} with {retrieval_args}",
        #     self.verbose,
        # )

        aggregated_data = get_and_print(
            lambda: pyearthtools.data.transforms.aggregation.over(method, dims)(
                self.index.series(
                    **retrieval_args,
                    transforms=transforms,
                ),
                **kwargs,
            ),
            f"Getting {method} with {retrieval_args}",
            self.verbose,
        )

        if save_pattern:
            if save_pattern.exists(variable_name, *format_class_name(self.index)):
                aggregate = save_pattern(variable_name, *format_class_name(self.index))
            else:
                LOG.debug(f"Could not find file for {variable_name}, calculating...")
                aggregate = aggregated_data()
                if not isinstance(aggregate, (xr.Dataset, xr.DataArray)):
                    raise TypeError(f"Aggregated data is a {type(aggregate)} must be an xarray object")

                xr.save_mfdataset(
                    tuple(aggregate[var].to_dataset() for var in aggregate.data_vars),
                    tuple(save_pattern.search(var) for var in aggregate.data_vars),
                )

        else:
            aggregate = aggregated_data()

        LOG.debug(f"Calculated aggregation, {method}-{variable_name}: {aggregate}")
        return aggregate

    def get_average(self, variable_name: Hashable):
        """Get data for average normalisation"""
        if self.override is not None:
            return open_file(self.override)

        self.check_init_args()
        return self.get_aggregation(variable_name, method="mean", dims=None)

    def get_deviation(self, variable_name: Hashable, spatial: bool = False) -> tuple[xr.Dataset, xr.Dataset]:
        """Get data for deviation normalisation"""
        if self.override is not None:
            if not isinstance(self.override, dict):
                raise ValueError(f"Cannot parse 'override': {self.override}.")
            return open_file(self.override["average"]), open_file(self.override["deviation"])

        self.check_init_args()
        return (
            self.get_aggregation(
                variable_name,
                method="mean",
                dims="time" if spatial else None,
                save_prefix="spatial" if spatial else "",
                dtype=float,
            ),
            self.get_aggregation(
                variable_name,
                method="std",
                dims="time" if spatial else None,
                save_prefix="spatial" if spatial else "",
                dtype=float,
            ),
        )

    def get_anomaly(self, variable_name: Hashable):
        """Get data for anomaly normalisation"""
        if self.override is not None:
            return open_file(self.override)

        self.check_init_args()
        return self.get_aggregation(variable_name, method="mean", dims="time", save_prefix="spatial")

    def get_range(self, variable_name: Hashable) -> dict:
        """
        Get Range of Data using [range][pyearthtools.data.AdvancedTimeIndex.range]

        Args:
            variable_name (str):
                Variable name to look for

        Returns:
            (dict):
                Dictionary filled with `max` and `min`
        """
        if self.override is not None:
            if isinstance(self.override, dict):
                if variable_name in self.override:
                    return self.override
                return {variable_name: self.override}
            return open_file(self.override)

        self.check_init_args()
        retrieval_args = self.retrieval_arguments
        transforms = retrieval_args.pop("transform", None)  # getattr(
        #     self.DataIndex, "base_transform", TransformCollection()
        # ) + retrieval_args.pop("transform", None)

        save_pattern = (
            pyearthtools.data.patterns.ArgumentExpansion(Path(self.cache_dir) / "range", extension=".json")
            if self.cache_dir
            else None
        )

        def find_range():
            data = self.index.series(**retrieval_args, transforms=transforms)

            max_values = data.max(skipna=True)
            min_values = data.min(skipna=True)

            return {var: {"max": max_values[var].data, "min": min_values[var].data} for var in max_values}

        range_data = get_and_print(
            lambda: find_range(),
            f"Getting range with {retrieval_args}",
            self.verbose,
        )

        if save_pattern:
            if save_pattern.search(variable_name, *format_class_name(self.index)).exists():
                range = save_pattern(variable_name, *format_class_name(self.index))
            else:
                range = range_data()

                def as_float(**kwargs):
                    return {key: float(value) for key, value in kwargs.items()}

                for key in range.keys():
                    save_pattern.save(
                        {key: as_float(**range[key])},
                        key,
                        *format_class_name(self.index),
                    )
        else:
            range = range_data()

        if not isinstance(range, dict):
            raise TypeError(f"Range data is a {type(range)} must be a dictionary")
        LOG.debug(f"Calculated aggregation, range-{variable_name}: {range[variable_name]}")

        return {variable_name: range[variable_name]}

    def _find_user_normaliser(self, key: str):
        if not "Normaliser" in key:
            raise AttributeError(f"{key!r} does not contain 'Normaliser', so is being ignored")
        try:
            return dynamic_import(key)(self)
        except AttributeError:
            raise AttributeError(f"{self.__class__} has no attribute {key}")

    @property
    def function(self) -> FunctionTransform:
        """
        Get Transform from user provided function for functional normalisation

        Returns:
            FunctionTransform: Transform to apply Function
        """
        raise NotImplementedError
        if self._function is None:
            raise ValueError("Cannot use function transform without a given function.\nTry giving `function` to init")
        return FunctionTransform(self._function)

    @property
    def none(norm_self) -> Transform:
        """
        Get a function to apply no normalisation

        Returns:
            Transform: Transform to apply no normalisation
        """

        class NoNormalisation(Transform):
            """Apply no Normalisation"""

            @property
            def _info_(self):
                return norm_self._info_

            def apply(self, dataset: xr.Dataset) -> xr.Dataset:
                return dataset

        return NoNormalisation()

    @abstractmethod
    def log(self):
        """
        Log Normalisation
        """
        raise NotImplementedError

    @abstractmethod
    def anomaly(self):
        """
        Anomaly based normalisation
        """
        raise NotImplementedError

    @abstractmethod
    def deviation(self):
        """
        Deviation based normalisation
        """
        raise NotImplementedError

    @abstractmethod
    def deviation_spatial(self):
        """
        Deviation based normalisation
        """
        raise NotImplementedError

    @abstractmethod
    def range(self):
        """
        Normalise using range to confine between 0 and 1
        """
        raise NotImplementedError

    @abstractmethod
    def manual_range(self, min, max):
        """
        Normalise using manual range to confine between 0 and 1
        """
        raise NotImplementedError

    def _per_variable_normaliser(norm_self, methods: dict, default: str = None) -> Transform:
        """
        From a methods dict assigning variable names to normalisation methods,
        get a Transform to apply each method correctly.

        Args:
            methods (dict): Dictionary assigning variable names to normalisation methods
            default (str, optional): A Default method to apply when not found in `methods`. Defaults to None.

        Raises:
            AttributeError: If variable not in dict and no default provided

        Returns:
            Transform: Transform to apply per variable normalisation
        """

        class PerVariableNormaliser(Transform):
            """Apply set normalisation approach on a per variable basis"""

            @property
            def _info_(self):
                return norm_self._info_

            def apply(self, dataset: xr.Dataset):
                for variable_name in dataset.data_vars:
                    if variable_name in methods:
                        dataset[variable_name] = norm_self(methods[variable_name])(dataset[variable_name].to_dataset())[
                            variable_name
                        ]
                    elif default or default is None:
                        dataset[variable_name] = norm_self(default)(dataset[variable_name].to_dataset())[variable_name]
                    else:
                        raise AttributeError(
                            f"{variable_name!r} not found in Normalisation Dict {methods} and no default was provided"
                        )
                return dataset

        return PerVariableNormaliser()

    def __call__(self, method: str | dict | tuple, default: str | None = None) -> Transform:
        """
        Get Transform to Normaliser or UnNormalise based on provided method

        Args:
            method (dict): Dictionary assigning variable names to normalisation methods
            default (str, optional): A Default method to apply when not found in `methods`. Defaults to None.

        Raises:
            KeyError: If an invalid method is provided.
            TypeError: If method is an invalid type

        Returns:
            Transform: Transform to apply normalisation
        """

        if default == "None":
            default = None

        if not method and default:
            method = default
        if method == "None" or method is None:
            return self.none

        if isinstance(method, str):
            if not hasattr(self, method):
                raise KeyError(
                    f"{self.__class__} has no method {method!r}."
                    "If this is a user specified function, ensure the name contains 'Normaliser'"
                )

            return getattr(self, method, getattr(self, default) if default else self.none)

        elif isinstance(method, dict):
            return self._per_variable_normaliser(method, default)

        elif isinstance(method, tuple):
            return self.manual_range(*method)
        raise TypeError(f"{type(method)!r} for 'method' not understood")

    def __repr__(self):
        return f"Normalisation Class waiting upon a request for a method, either call with a method or use property."

    def __del__(self):
        if hasattr(self, "temp_dir"):
            self.temp_dir.cleanup()
