# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.


import math
from typing import Literal, Union

import numpy as np
from edit.pipeline.filters import Filter, PipelineFilterException


class NumpyFilter(Filter):
    """Numpy Filters"""

    _override_interface = ["Delayed", "Serial"]

    def __init__(self):
        super().__init__(
            split_tuples=True,
            recursively_split_tuples=True,
            recognised_types=np.ndarray,
        )


class DropAnyNan(NumpyFilter):
    """
    Filter to drop any data with nans when iterating.

    Used to remove any bad data or data that is masked out.
    """

    _interface_kwargs = {"Delayed": {"name": "DropAnyNan"}}

    def __init__(self) -> None:
        """Drop data with any nans"""
        super().__init__()
        self.record_initialisation()

    def filter(self, sample: np.ndarray):
        """Check if any of the sample is nan

        Args:
            sample (np.ndarray):
                Sample to check
        Returns:
            (bool):
                If sample contains nan's
        """
        if not bool(np.array(list(np.isnan(sample))).any()):
            raise PipelineFilterException(sample, f"Data contained nan's.")


class DropAllNan(NumpyFilter):
    """
    Filter to drop any data if all nans.

    Used to remove any bad data or data that is masked out.
    """

    _interface_kwargs = {"Delayed": {"name": "DropAllNan"}}

    def __init__(self) -> None:
        """Drop data with any nans"""
        super().__init__()
        self.record_initialisation()

    def filter(self, sample: np.ndarray):
        """Check if all of the sample is nan

        Args:
            sample (np.ndarray):
                Sample to check
        Returns:
            (bool):
                If sample contains nan's
        """
        if not bool(np.array(list(np.isnan(sample))).all()):
            raise PipelineFilterException(sample, f"Data contained all nan's.")


class DropValue(NumpyFilter):
    """
    Filter to drop data containing more than a given percentage of a value.

    Can be used to trim out invalid data
    """

    _interface_kwargs = {"Delayed": {"name": "DropValue"}}

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

    def filter(self, sample: np.ndarray):
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

    _interface_kwargs = {"Delayed": {"name": "ShapeFilter"}}

    def __init__(self, shape: tuple[Union[tuple[int, ...], int], ...], split_tuples: bool = False) -> None:
        """
        Drop Data if shape does not match expected

        Args:
            shape tuple[Union[tuple[int, ...], int]):
                Shape to match, either tuple of shapes for tupled data or direct shape
            split_tuples (bool, optional):
                Whether to split tuples, if `True`, `shape` should not be a tuple of tuples
        """
        super().__init__(split_tuples=split_tuples, recognised_types=np.ndarray)
        self.record_initialisation()

        self._shape = shape

    def _find_shape(self, data: Union[tuple[np.ndarray, ...], np.ndarray]) -> tuple[Union[tuple, int], ...]:
        if isinstance(data, tuple):
            return tuple(map(self._find_shape, data))
        return data.shape

    def check_shape(self, sample: Union[tuple[np.ndarray, ...], np.ndarray]):
        if isinstance(sample, (list, tuple)):
            if not isinstance(self._shape, (list, tuple)) and len(self._shape) == len(sample):
                raise RuntimeError(
                    f"If sample is tuple, shape must also be, and of the same length. {self._shape} != {tuple(self._find_shape(i) for i in sample)}"
                )

        if not self._find_shape(sample) == self._shape:
            raise PipelineFilterException(
                sample, f"Shapes were found not to be the same.\n{self._find_shape(sample)} != {self._shape}"
            )
