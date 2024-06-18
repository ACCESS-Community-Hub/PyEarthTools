# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from typing import TypeVar, Union, Optional, Any

import xarray as xr

from edit.pipeline_V2.branching.split import Spliter

T = TypeVar("T", xr.Dataset, xr.DataArray)


class OnVariables(Spliter):
    """Split xarray object's on variables"""

    def __init__(
        self,
        variables: Union[tuple[Union[str, tuple[str, ...], list[str]], ...], list[str]],
        merge_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            recognised_types=(xr.DataArray, xr.Dataset),
            recursively_split_tuples=True,
        )
        self.record_initialisation()

        self._variables = variables
        self._merge_kwargs = merge_kwargs

    def split(self, sample: xr.Dataset) -> tuple[xr.Dataset, ...]:
        """Split sample"""

        subsets = []
        for var in self._variables:
            if any(map(lambda x: x not in sample, (var,) if not isinstance(var, (tuple, list)) else var)):
                raise ValueError(
                    f"Could not split on {var}, as it was not found in dataset. Found {list(sample.data_vars)}."
                )
            subsets.append(sample[list(var) if isinstance(var, (tuple, list)) else [var]])
        return tuple(subsets)

    def join(self, sample: tuple[Union[xr.Dataset, xr.DataArray], ...]) -> xr.Dataset:
        """Join sample"""
        return xr.merge(sample, **(self._merge_kwargs or {}))


class OnCoordinate(Spliter):
    """Split xarray object on coordinate"""

    def __init__(
        self,
        coordinate: str,
        merge_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            recursively_split_tuples=True,
            recognised_types=(xr.DataArray, xr.Dataset),
        )
        self.record_initialisation()

        self.coordinate = coordinate
        self._merge_kwargs = merge_kwargs

    def split(self, sample: T) -> tuple[T, ...]:
        return tuple(sample.sel(**{self.coordinate: i}) for i in sample.coords[self.coordinate])

    def undo(self, sample: tuple[T, ...]) -> xr.Dataset:
        return xr.merge(sample, **(self._merge_kwargs or {}))
