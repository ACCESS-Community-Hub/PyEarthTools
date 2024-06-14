# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from typing import Hashable, TypeVar, Union

import xarray as xr

import edit.data
from edit.pipeline_V2.operation import Operation

T = TypeVar("T", xr.Dataset, xr.DataArray)


class Dimensions(Operation):
    def __init__(
        self,
        dimensions: Union[str, list[str]],
        append: bool = True,
        preserve_order: bool = False,
    ):
        """
        Operation to reorder Dimensions of an [xarray][xarray] object.

        Not all dims have to be supplied, will automatically add remaining dims,
        or if append == False, prepend extra dims.

        Args:
            dimensions (Union[str, list[str]]):
                Specified order of dimensions to tranpose dataset to
            append (bool, optional):
                Append extra dims, if false, prepend dims. Defaults to True.
            preserve_order (bool, optional):
                Whether to preserve the order of dims or on `undo`, also set to dimensions order.
                Defaults to False.
        """
        super().__init__(
            split_tuples=True,
            recursively_split_tuples=True,
            recognised_types=(xr.Dataset, xr.DataArray),
        )
        self.record_initialisation()

        self.dimensions = dimensions if isinstance(dimensions, (list, tuple)) else [dimensions]
        self.append = append
        self.preserve_order = preserve_order

        self._incoming_dims = None

        self.__doc__ = "Reorder Dimensions"

    def apply_func(self, sample: T) -> T:
        dims = sample.dims
        self._incoming_dims = list(dims)

        dims = set(dims).difference(set(self.dimensions))

        if self.append:
            dims = [*self.dimensions, *dims]
        else:
            dims = [*dims, *self.dimensions]

        if self.preserve_order:
            self._incoming_dims = dims

        return sample.transpose(*dims, missing_dims="ignore")

    def undo_func(self, sample: T) -> T:
        if self._incoming_dims:
            return sample.transpose(*self._incoming_dims, missing_dims="ignore")
        return sample


class CoordinateFlatten(Operation):
    """Flatten and Expand on a coordinate"""

    def __init__(self, coordinate: Union[Hashable, list[Hashable]], *coords: Hashable, skip_missing: bool = False):
        """
        Flatten and expand on coordinate/s

        Args:
            coordinate (Union[Hashable,list[Hashable]]):
                Coordinate to flatten and expand on.
            skip_missing (bool, optional):
                Whether to skip data without the dims. Defaults to False
        """
        super().__init__(
            split_tuples=True,
            recursively_split_tuples=True,
            recognised_types=(xr.Dataset, xr.DataArray),
        )
        self.record_initialisation()

        coordinate = [coordinate, *coords] if not isinstance(coordinate, (list, tuple)) else [*coordinate, *coords]
        self.coords = coordinate
        self._skip_missing = skip_missing

    def apply_func(self, ds):
        return edit.data.transforms.coordinates.flatten(self.coords, skip_missing=self._skip_missing)(ds)

    def undo_func(self, ds):
        return edit.data.transforms.coordinates.expand(self.coords)(ds)
