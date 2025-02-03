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
Remapping Operations.
"""

from abc import abstractmethod, ABCMeta
from typing import Type, TypeVar
import xarray as xr

from pyearthtools.pipeline import Operation

XR_TYPE = TypeVar("XR_TYPE", xr.Dataset, xr.DataArray)


class BaseRemap(Operation, metaclass=ABCMeta):
    """
    Base class for remappers.

    Child class must implement `remap` and `inverse_remap`
    """

    _override_interface = "Serial"

    def __init__(
        self,
        *,
        split_tuples: bool = True,
        recursively_split_tuples: bool = True,
        recognised_types: tuple[Type, ...] = (xr.Dataset, xr.DataArray),
    ):
        super().__init__(
            split_tuples=split_tuples,
            recursively_split_tuples=recursively_split_tuples,
            recognised_types=recognised_types,
        )

    def apply_func(self, sample):
        return self.remap(sample)

    def undo_func(self, sample):
        return self.inverse_remap(sample)

    @abstractmethod
    def remap(self, sample: XR_TYPE) -> XR_TYPE:
        """
        Forward mapping operation. Must be defined in subclasses.
        """
        raise NotImplementedError()

    @abstractmethod
    def inverse_remap(self, sample: XR_TYPE) -> XR_TYPE:
        """
        Inverse mapping operation. Must be defined in subclasses.
        """
        raise NotImplementedError()
