# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from typing import TypeVar, Optional, Literal

import xarray as xr

from edit.pipeline.operation import Operation

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
                Chunk dictionary. coord: size. Defaults to None.
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
