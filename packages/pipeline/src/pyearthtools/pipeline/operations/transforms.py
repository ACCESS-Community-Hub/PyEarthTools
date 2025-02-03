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


from typing import Any, TypeVar, Union, Optional

import xarray as xr

import pyearthtools.data
from pyearthtools.pipeline.operation import Operation

T = TypeVar("T", xr.Dataset, xr.DataArray)
TRANSFORM_TYPE = Union[pyearthtools.data.Transform, pyearthtools.data.TransformCollection]


class Transforms(Operation):
    """
    Run `pyearthtools.data.Transforms` within a `Pipeline`.
    """

    _override_interface = "Serial"

    def __init__(
        self,
        transforms: Optional[TRANSFORM_TYPE] = None,
        apply: Optional[TRANSFORM_TYPE] = None,
        undo: Optional[TRANSFORM_TYPE] = None,
    ):
        """
        Run `Transforms`

        If `transforms` given will run on both functions first, and then if also given `apply` and `undo` respectively.

        Args:
            transforms (Optional[TRANSFORM_TYPE], optional):
                Transforms to run on both `apply` and `undo`. Defaults to None.
            apply (Optional[TRANSFORM_TYPE], optional):
                Transforms to run on `apply`. Defaults to None.
            undo (Optional[TRANSFORM_TYPE], optional):
                Transforms to run on `undo`. Defaults to None.
        """
        super().__init__(split_tuples=True, recursively_split_tuples=True)
        self.record_initialisation()

        self._transforms = pyearthtools.data.TransformCollection() + (transforms if transforms is not None else [])
        self._apply_transforms = pyearthtools.data.TransformCollection() + (apply if apply is not None else [])
        self._undo_transforms = pyearthtools.data.TransformCollection() + (undo if undo is not None else [])

    def apply_func(self, sample: T) -> T:
        sample = self._transforms(sample)
        return self._apply_transforms(sample)

    def undo_func(self, sample: T) -> T:
        sample = self._transforms(sample)
        return self._undo_transforms(sample)
