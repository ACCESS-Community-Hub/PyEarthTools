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


from typing import Hashable, TypeVar, Union

import xarray as xr
import numpy as np

import pyearthtools.data
from pyearthtools.data.transforms.coordinates import Drop
from pyearthtools.pipeline.operation import Operation

from pyearthtools.data.transforms.attributes import SetType
from pyearthtools.utils.decorators import BackwardsCompatibility

T = TypeVar("T", xr.Dataset, xr.DataArray)


class Dimensions(Operation):
    """
    Reorder dimensions
    """

    _override_interface = "Serial"

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
                Specified order of dimensions to transpose dataset to
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


def weak_cast_to_int(value):
    """
    Basically, turns integer floats to int types, otherwise
    does nothing. Used in CoordinateFlatten.
    """
    try:
        if int(value) == value:
            value = int(value)
    except Exception:
        pass
    return value


class CoordinateFlatten(Operation):
    """Flatten a coordinate in a dataset into separate variables."""

    _override_interface = "Serial"

    def __init__(self, coordinate: Hashable, skip_missing: bool = False):
        """
        Flatten a coordinate in an xarray Dataset, putting the data at each value of the coordinate into a separate
        data variable.

        The output data variables will be named "<old variable name><value of coordinate>". For example, if the input
        Dataset has a variable "t" and it is flattened along the coordinate "pressure_level" which has values
        [100, 200, 500], then the output Dataset will have variables called t100, t200 and t500.

        Args:
            coordinate (Hashable):
                Coordinate to flatten and expand on.
            skip_missing (bool, optional):
                Whether to skip data that does not have any of the listed coordinates. If True, will return such data
                unchanged. Defaults to False.

        Raises:
            ValueError:
                If coordinate not found in the dataset and skip_missing==False.
        """
        super().__init__(
            split_tuples=True,
            recursively_split_tuples=True,
            recognised_types=(xr.Dataset, xr.DataArray),
        )
        self.record_initialisation()

        self._coordinate = coordinate
        self._skip_missing = skip_missing

    def apply_func(self, dataset: xr.Dataset) -> xr.Dataset:
        discovered_coord = list(set(self._coordinate).intersection(set(dataset.coords)))

        if len(discovered_coord) == 0:
            if self._skip_missing:
                return dataset

            raise ValueError(
                f"{self._coordinate} could not be found in dataset with coordinates {list(dataset.coords)}.\n"
                "Set 'skip_missing' to True to skip this."
            )

        discovered_coord = str(discovered_coord[0])

        coords = dataset.coords
        new_ds = xr.Dataset(coords={co: v for co, v in coords.items() if not co == discovered_coord})
        new_ds.attrs.update(
            {f"{discovered_coord}-dtype": str(dataset[discovered_coord].encoding.get("dtype", "int32"))}
        )

        for var in dataset:
            if discovered_coord not in dataset[var].coords:
                new_ds[var] = dataset[var]
                continue

            coord_size = dataset[var][discovered_coord].values
            coord_size = coord_size if isinstance(coord_size, np.ndarray) else np.array(coord_size)

            if coord_size.size == 1 and False:
                coord_val = weak_cast_to_int(dataset[var][discovered_coord].values)
                new_ds[f"{var}{coord_val}"] = Drop(discovered_coord, ignore_missing=True)(dataset[var])

            else:
                for coord_val in dataset[discovered_coord]:
                    coord_val = weak_cast_to_int(coord_val.values.item())

                    selected = dataset[var].sel(**{discovered_coord: coord_val})  # type: ignore
                    selected = selected.drop_vars(discovered_coord)  # type: ignore
                    selected.attrs.update(**{discovered_coord: coord_val})

                    new_ds[f"{var}{coord_val}"] = selected
        return new_ds

    def undo_func(self, ds):
        return pyearthtools.pipeline.operations.xarray.reshape.coordinate_expand(self._coordinate)(ds)


@BackwardsCompatibility(CoordinateFlatten)
def coordinate_flatten(*args, **kwargs) -> Operation: ...


class CoordinateExpand(Operation):
    """Inverse operation to `CoordinateFlatten`"""

    def __init__(self, coordinate: Hashable):
        """
        Inverse operation to [flatten][pyearthtools.pipeline.operations.xarray.reshape.CoordinateFlatten]

        Will find flattened variables and regroup them upon the extra coordinate

        Args:
            coordinate (Hashable):
                Coordinate to unflatten.
        """
        super().__init__()
        self.record_initialisation()

        if not isinstance(coordinate, (list, tuple)):
            coordinate = (coordinate,)

        self._coordinate = coordinate

    def apply_func(self, dataset: xr.Dataset) -> xr.Dataset | xr.DataArray:
        dataset = type(dataset)(dataset)

        for coord in self._coordinate:
            dtype = dataset.attrs.get(f"{coord}-dtype", "int32")
            components = []
            for var in list(dataset.data_vars):
                var_data = dataset[var]
                if coord in var_data.attrs:
                    value = var_data.attrs.pop(coord)
                    var_data = (
                        var_data.to_dataset(name=var.replace(str(value), ""))
                        .assign_coords(**{coord: [value]})
                        .set_coords(coord)
                    )
                components.append(var_data)

            dataset = xr.combine_by_coords(components)  # type: ignore
            dataset = SetType(**{str(coord): dtype})(dataset)

            ## Add stored encoding if there
            if f"{coord}-dtype" in dataset.attrs:
                dtype = dataset.attrs.pop(f"{coord}-dtype")
                dataset[coord].encoding.update(dtype=dtype)

        return dataset

    def undo_func(self, ds):
        return pyearthtools.pipeline.operations.xarray.reshape.coordinate_flatten(self._coordinate)(ds)


@BackwardsCompatibility(CoordinateExpand)
def coordinate_expand(*args, **kwargs) -> Operation: ...
