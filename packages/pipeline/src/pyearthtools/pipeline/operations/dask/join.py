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


# type: ignore[reportPrivateImportUsage]

from itertools import accumulate
from typing import Optional, Any

import dask.array as da

from pyearthtools.pipeline.branching.join import Joiner
from pyearthtools.pipeline.operations.dask.dask import DaskOperation


class Stack(Joiner, DaskOperation):
    """
    Stack a tuple of da.Array's
    """

    _override_interface = ["Serial"]
    _numpy_counterpart = "join.Stack"

    def __init__(self, axis: Optional[int] = 0):
        super().__init__()
        self.record_initialisation()
        self.axis = axis

    def join(self, sample: tuple[Any, ...]) -> da.Array:
        """Join sample"""
        return da.stack(sample, self.axis)  # type: ignore

    def unjoin(self, sample: Any) -> tuple:
        """Unstacks a stacked sample"""
        # move the stacked axis to the zeroth axis and convert to tuple
        return tuple(da.moveaxis(sample, self.axis, 0))


class VStack(Joiner, DaskOperation):
    """
    Vertically Stack a tuple of da.Array's
    """

    _override_interface = ["Serial"]
    _numpy_counterpart = "join.VStack"

    def __init__(self):
        super().__init__()
        self.record_initialisation()
        self.offsets = None

    def join(self, sample: tuple[Any, ...]) -> da.Array:
        """Join sample"""
        self.offsets = tuple(accumulate(arr.shape[0] for arr in sample[:-1]))
        return da.vstack(
            sample,
        )  # type: ignore

    def unjoin(self, sample: Any) -> tuple:
        start = (0,) + self.offsets
        ends = self.offsets + (sample.shape[0],)
        return tuple(sample[start:end] for start, end in zip(start, ends, strict=True))


class HStack(Joiner, DaskOperation):
    """
    Horizontally Stack a tuple of da.Array's
    """

    _override_interface = ["Serial"]
    _numpy_counterpart = "join.HStack"

    def __init__(self):
        super().__init__()
        self.record_initialisation()
        self.offsets = None

    def join(self, sample: tuple[Any, ...]) -> da.Array:
        """Join sample"""
        self.offsets = tuple(accumulate(arr.shape[1] for arr in sample[:-1]))
        return da.hstack(
            sample,
        )  # type: ignore

    def unjoin(self, sample: Any) -> tuple:
        start = (0,) + self.offsets
        ends = self.offsets + (sample.shape[1],)
        return tuple(sample[:, start:end, ...] for start, end in zip(start, ends, strict=True))


class Concatenate(Joiner, DaskOperation):
    """
    Concatenate a tuple of da.Array's
    """

    _override_interface = ["Serial"]
    _numpy_counterpart = "join.Concatenate"

    def __init__(self, axis: Optional[int] = None):
        super().__init__()
        self.record_initialisation()
        self.axis = axis
        self.offsets = None

    def join(self, sample: tuple[Any, ...]) -> da.Array:
        """Join sample"""
        self.offsets = tuple(accumulate(arr.shape[self.axis] for arr in sample[:-1]))
        return da.concatenate(sample, self.axis)  # type: ignore

    def unjoin(self, sample: Any) -> tuple:
        start = (0,) + self.offsets
        ends = self.offsets + (sample.shape[self.axis],)
        return tuple(da.take(sample, slice(start, end), self.axis) for start, end in zip(start, ends, strict=True))
