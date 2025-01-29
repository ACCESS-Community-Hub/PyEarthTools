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


from typing import TypeVar, Union, Optional, Any

import xarray as xr

from pyearthtools.pipeline.branching.join import Joiner

T = TypeVar("T", xr.Dataset, xr.DataArray)


class Merge(Joiner):
    """
    Merge a tuple of xarray object's.

    Currently cannot undo this operation
    """

    _override_interface = "Serial"

    def __init__(self, merge_kwargs: Optional[dict[str, Any]] = None):
        super().__init__()
        self.record_initialisation()
        self._merge_kwargs = merge_kwargs

    def join(self, sample: tuple[Union[xr.Dataset, xr.DataArray], ...]) -> xr.Dataset:
        """Join sample"""
        return xr.merge(sample, **(self._merge_kwargs or {}))

    def unjoin(self, sample: Any) -> tuple:
        return super().unjoin(sample)


class Concatenate(Joiner):
    """
    Concatenate a tuple of xarray object's

    Currently cannot undo this operation
    """

    _override_interface = "Serial"

    def __init__(self, concat_dim: str, concat_kwargs: Optional[dict[str, Any]] = None):
        super().__init__()
        self.record_initialisation()
        self._concat_dim = concat_dim

        if concat_kwargs:
            concat_kwargs.pop("dim", None)

        self._concat_kwargs = concat_kwargs

    def join(self, sample: tuple[T, ...]) -> T:
        """Concat sample"""
        return xr.concat(sample, dim=self._concat_dim, **(self._concat_kwargs or {}))  # type: ignore

    def unjoin(self, sample: Any) -> tuple:
        return super().unjoin(sample)
