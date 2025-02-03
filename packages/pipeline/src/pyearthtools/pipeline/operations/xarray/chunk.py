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


from typing import TypeVar, Optional, Literal

import xarray as xr

from pyearthtools.pipeline.operation import Operation

T = TypeVar("T", xr.Dataset, xr.DataArray)


class Chunk(Operation):
    """ReChunk xarray object"""

    _override_interface = "Serial"

    def __init__(
        self,
        chunk: Optional[dict[str, int]] = None,
        operation: Literal["apply", "undo", "both"] = "apply",
        **extra_chunk_kwargs: int,
    ):
        """
        ReChunk xarray object

        Args:
            chunk (Optional[dict[str, int]], optional):
                Chunk dictionary. `coord: size`. Defaults to None.
            operation (Literal['apply', 'undo', 'both']):
                When to apply rechunking. Defaults to 'apply'.
            **extra_chunk_kwargs (int):
                Kwarg form of `chunk`.
        """
        super().__init__(
            split_tuples=True,
            recognised_types=(xr.Dataset, xr.DataArray),
            operation=operation,
        )
        self.record_initialisation()
        chunk = chunk or {}
        chunk.update((extra_chunk_kwargs))
        self._chunk = chunk

    def apply_func(self, sample: T) -> T:
        return sample.chunk(**self._chunk)  # type: ignore

    def undo_func(self, sample: T) -> T:
        return self.apply_func(sample)
