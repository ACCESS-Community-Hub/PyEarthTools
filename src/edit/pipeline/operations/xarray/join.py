# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from typing import TypeVar, Union, Optional, Any

import xarray as xr

from edit.pipeline.branching.join import Joiner

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
