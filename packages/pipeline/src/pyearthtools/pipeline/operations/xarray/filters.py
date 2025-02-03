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


from typing import Literal, Optional, TypeVar, Union

import numpy as np
import xarray as xr

import math

from pyearthtools.pipeline.filters import Filter, PipelineFilterException

T = TypeVar("T", xr.Dataset, xr.DataArray)


class XarrayFilter(Filter):
    """Xarray Filters"""

    _override_interface = "Serial"

    def __init__(self):
        super().__init__(
            split_tuples=True,
            recursively_split_tuples=True,
            recognised_types=(xr.Dataset, xr.DataArray),
        )


class DropAnyNan(XarrayFilter):
    """
    Filter to drop any data with nans when iterating.

    Used to remove any bad data or data that is masked out.
    """

    def __init__(self, variables: Optional[list] = None) -> None:
        """Drop data with any nans

        Args:
            variables (list, optional):
                Subset of variables to check.
                Defaults to None.
        """
        super().__init__()
        self.record_initialisation()

        self.variables = variables

    def _check(self, sample: xr.Dataset):
        """Check if any of the sample is nan

        Args:
            sample (xr.Dataset):
                Sample to check
        Returns:
            (bool):
                If sample contains nan's
        """
        if self.variables:
            sample = sample[self.variables]

        if not bool(np.array(list(np.isnan(sample).values())).any()):
            raise PipelineFilterException(sample, f"Data contained nan's.")


class DropAllNan(XarrayFilter):
    """
    Filter to drop any data with all nans when iterating.

    Used to remove any bad data or data that is masked out.
    """

    def __init__(self, variables: Optional[list] = None) -> None:
        """Drop data with all nans

        Args:
            variables (list, optional):
                Subset of variables to check.
                Defaults to None.
        """
        super().__init__()
        self.record_initialisation()

        self.variables = variables

    def _check(self, sample: xr.Dataset):
        """Check if all of the sample is nan

        Args:
            sample (xr.Dataset):
                Sample to check
        Returns:
            (bool):
                If sample contains nan's
        """
        if self.variables:
            sample = sample[self.variables]

        if not bool(np.array(list(np.isnan(sample).values())).all()):
            raise PipelineFilterException(sample, f"Data contained all nan's.")


class DropValue(XarrayFilter):
    """
    Filter to drop data containing more than a given percentage of a value.

    Can be used to trim out invalid data
    """

    def __init__(self, value: Union[float, Literal["nan"]], percentage: float) -> None:
        """Drop Data if number of elements equal to value are greater than percentage when iterating.

        Args:
            value (Union[float, Literal["nan"]]):
                Value to search for. Can be nan or 'nan'.
            percentage (float):
                Percentage of `value` of which an exceedance drops data
        """
        super().__init__()
        self.record_initialisation()

        if isinstance(value, str) and value == "nan":
            value = np.nan

        self._value = value
        self._percentage = percentage

    def filter(self, sample: T):
        """Check if all of the sample is nan

        Args:
            sample (np.ndarray):
                Sample to check
        Returns:
            (bool):
                If sample contains nan's
        """
        if np.isnan(self._value):
            function = lambda x: ((np.count_nonzero(np.isnan(x)) / math.prod(x.shape)) * 100) >= self._percentage
        else:
            function = lambda x: ((np.count_nonzero(x == self._value) / math.prod(x.shape)) * 100) >= self._percentage

        if not function(sample):
            raise PipelineFilterException(sample, f"Data contained more than {self._percentage}% of {self._value}.")


class Shape(Filter):
    """
    Filter to drop data of incorrect shape

    Used to ensure that incoming data is of the correct shape for later steps
    """

    _override_interface = "Serial"

    def __init__(self, shape: tuple[Union[tuple[int, ...], int], ...], split_tuples: bool = False) -> None:
        """
        Drop Data if shape does not match expected

        Args:
            shape tuple[Union[tuple[int, ...], int]):
                Shape to match, either tuple of shapes for tupled data or direct shape
            split_tuples (bool, optional):
                Whether to split tuples, if `True`, `shape` should not be a tuple of tuples
        """
        super().__init__(split_tuples=split_tuples, recognised_types=(xr.Dataset, xr.DataArray))
        self.record_initialisation()

        self._shape = shape

    def _find_shape(self, data: T) -> tuple[int, ...]:
        if isinstance(data, xr.Dataset):
            shape = (
                len(list(data.data_vars)),
                *data[list(data.data_vars)[0]].shape,
            )
        elif isinstance(data, xr.DataArray):
            shape = data.shape
        else:
            raise TypeError(f"Unable to find shape of {data!r}")
        return shape

    def filter(self, sample: Union[tuple[T, ...], T]):
        if isinstance(sample, (list, tuple)):
            if not isinstance(self._shape, (list, tuple)) and len(self._shape) == len(sample):
                raise RuntimeError(
                    f"If sample is tuple, shape must also be, and of the same length. {self._shape} != {tuple(self._find_shape(i) for i in sample)}"
                )

            for i in range(len(sample)):
                if not tuple(self._find_shape(sample[i])) == tuple(self._shape[i]):  # type: ignore
                    raise PipelineFilterException(sample, f"Shapes were found not to be the same. At tuple index {i}.\n{tuple(self._find_shape(sample[i]))} != {tuple(self._shape[i])}")  # type: ignore
            return True

        if not self._find_shape(sample) == self._shape:
            raise PipelineFilterException(
                sample, f"Shapes were found not to be the same.\n{self._find_shape(sample)} != {self._shape}"
            )
