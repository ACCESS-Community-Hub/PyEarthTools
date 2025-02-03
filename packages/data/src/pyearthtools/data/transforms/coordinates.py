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


from __future__ import annotations

import warnings
from typing import Any, Hashable, Literal, Iterable
import logging

import xarray as xr
import numpy as np


import pyearthtools.data

from pyearthtools.data.transforms.transform import Transform, TransformCollection
from pyearthtools.data.warnings import pyearthtoolsDataWarning
from pyearthtools.data.exceptions import DataNotFoundError

from pyearthtools.utils.decorators import BackwardsCompatibility

DASK_IMPORTED = False
try:
    import dask

    DASK_IMPORTED = True
except ImportError:
    DASK_IMPORTED = False


LOG = logging.getLogger("pyearthtools.data")

VALID_COORDINATE_DEFINITIONS = Literal["-180-180", "0-360"]


def get_longitude(data: xr.Dataset | xr.DataArray, transform: bool = True) -> VALID_COORDINATE_DEFINITIONS | Transform:
    """
    From a given data source, attempt to identify the orientation of the `longitude` coordinate.

    Either '0-360' or '-180-180'

    Args:
        data (xr.Dataset | xr.DataArray):
            Data to check
        transform (bool, optional):
            Whether to return a `Transform` to set to the same orientation. Defaults to True.

    Raises:
        ValueError:
            If unable to identify the `longitude` coordinate orientation

    Returns:
        (str | Transform):
            Either str of orientation or Transform to set longitude of a data source to the same as `data`
            Depends on `transform` bool state.
    """
    if "longitude" not in data.coords:
        raise ValueError(f"Cannot get longitude from data, has coords {data.coords}.")

    def _return(
        coord_orientation: VALID_COORDINATE_DEFINITIONS,
    ) -> VALID_COORDINATE_DEFINITIONS | Transform:
        if not transform:
            return coord_orientation
        return standard_longitude(coord_orientation)

    if any(data.longitude.values > 180):
        return _return("0-360")
    elif any(data.longitude < 0):
        return _return("-180-180")

    raise ValueError(f"Could not identify longitude coordinate from data. {data.longitude}")
    LOG.debug(f"Could not identify longitude coordinate from data. {data.longitude}")


class StandardLongitude(Transform):
    """Standardise format of longitude."""

    def __init__(self, type: VALID_COORDINATE_DEFINITIONS = "-180-180", longitude_name="longitude"):
        """
        Standardise format of longitude.

        Shifts the longitude coordinate to that of the specified. Must be in ["-180-180", "0-360"]

        Args:
            type (VALID_COORDINATE_DEFINITIONS): Longitude Specification. Defaults to "-180-180".

        """

        super().__init__()
        self.record_initialisation()

        valid_types = ["-180-180", "0-360"]
        if type not in valid_types:
            raise KeyError(f"Invalid `type` passed, must be one of {valid_types} not {type}")
        self._type = type
        self._longitude_name = longitude_name

    @property
    def _info_(self):
        return dict(type=self._type)

    def apply(self, dataset):
        if self._type == "0-360":

            def _standardise(dataset):
                if not any(dataset[self._longitude_name] < 0):
                    return dataset
                func = lambda x: x % 360
                dataset = dataset.assign_coords({self._longitude_name: func(dataset[self._longitude_name])})
                return dataset.sortby(self._longitude_name)

        else:

            def _standardise(dataset):
                if not any(dataset[self._longitude_name] > 180):
                    return dataset
                func = lambda x: ((x + 180) % 360) - 180
                # (180 - abs(x - 180)) * np.sign((x - 180)) * -1
                dataset = dataset.assign_coords({self._longitude_name: func(dataset[self._longitude_name])})
                return dataset.sortby(self._longitude_name)

        if self._longitude_name not in dataset.coords:
            warnings.warn(
                f"Could not move longitude to {self._type}, '{self._longitude_name}' is not in coords.",
                pyearthtoolsDataWarning,
            )

        if DASK_IMPORTED:
            with dask.config.set(**{"array.slicing.split_large_chunks": True}):  # type: ignore
                dataset = _standardise(dataset)
        else:
            dataset = _standardise(dataset)

        return dataset


@BackwardsCompatibility(StandardLongitude)
def standard_longitude(type: VALID_COORDINATE_DEFINITIONS = "-180-180") -> Transform: ...


class ReIndex(Transform):
    """Reindex Coordinates"""

    def __init__(
        self, coordinates: dict[str, Literal["reversed", "sorted"] | Iterable] | xr.Coordinates | None = None, **coords
    ):
        """
        Reindex coordinates

        Can be sorted, or in set list

        Args:
            coordinates (dict[str, Literal['reversed','sorted'] | Iterable | xr.Coordinates] | None, optional):
                Coordinate to reindex, and Iterable to reindex at.
                If 'reversed' or 'sorted', take current coord and sort.
                If `xr.Coordinates`, use any coordinates with len > 1.
                Defaults to None.

        """
        super().__init__()
        self.record_initialisation()

        if coordinates is None:
            coordinates = {}

        if isinstance(coordinates, xr.Coordinates):
            coordinates = {
                str(coord): list(coordinates[coord].values)
                for coord in coordinates
                if len(np.atleast_1d(coordinates[coord].values)) > 1
            }

        coordinates = dict(coordinates)
        coordinates.update(coords)

        if not coordinates:
            raise ValueError(f"No coordinates to reindex at, must be given either with `coordinates` or `kwargs`.")
        self._coordinates = coordinates

    @property
    def _info_(self):
        return dict(**self._coordinates)

    def apply(self, dataset: xr.Dataset):
        for coord, index_op in self._coordinates.items():
            if not coord in dataset.coords:
                continue

            if isinstance(index_op, str):
                new_coord = sorted(dataset[coord].values, reverse=index_op == "reversed")
            elif isinstance(index_op, Iterable):
                new_coord = index_op
            else:
                raise TypeError(f"Cannot parse index {index_op!r}, must be string or Iterable.")

            dataset = dataset.reindex({coord: new_coord})

        return dataset


@BackwardsCompatibility(ReIndex)
def reindex(*args, **kwargs) -> Transform: ...


class StandardCoordinateNames(Transform):
    """Convert xr.Dataset Coordinate Names into Standard Naming Scheme"""

    def __init__(self, replacement_dictionary: dict | None = None, **repl_kwargs):
        """
        Convert xr.Dataset Coordinate Names into Standard Naming Scheme

        Args:
            replacement_dictionary (dict | None, optional):
                Dictionary assigning name replacements [old: new].
                One of replacement_dictionary or repl_kwargs must be provided. Defaults to None.
            **repl_kwargs (dict, optional):
                Kwarg version of replacement_dictionary

        """

        super().__init__()
        self.record_initialisation()

        if replacement_dictionary is None:
            replacement_dictionary = {}

        replacement_dictionary.update(repl_kwargs)
        self._replacements = replacement_dictionary

    # @property
    # def _info_(self):
    #     return dict(**self._replacements)

    def apply(self, dataset: xr.Dataset):
        for correctname, falsenames in self._replacements.items():
            for falsename in set(falsenames) & set(dataset.dims):
                dataset = dataset.rename({falsename: correctname})

            for falsename in set(falsenames) & set(dataset.coords):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # TODO  UserWarning Raised below
                    dataset = dataset.rename({falsename: correctname})
        return dataset


@BackwardsCompatibility(StandardCoordinateNames)
def force_standard_coordinate_names(*args, **kwargs) -> Transform: ...


class Select(Transform):
    """Select on Coordinates"""

    def __init__(
        self,
        indexers: dict[str, Any] | None = None,
        *,
        ignore_missing: bool = False,
        tolerance: float | None = None,
        isel: bool = False,
        **indexers_kwargs,
    ):
        """
        Select values on coordinates

        Args:
            indexers (dict[str, Any] | None, optional):
                A dict with keys matching dimensions and values
                One of indexers or indexers_kwargs must be provided. Defaults to None.
            **indexers_kwargs (dict):
                Index keyword arguments
            ignore_missing (bool, optional):
                Ignore coordinates not in dataset. Defaults to False
            tolerance (float | None, optional):
                Tolerance for selection. Defaults to None.
            isel (bool, optional):
                Whether to use isel. Defaults to False.
        """
        super().__init__()
        self.record_initialisation()

        if indexers is None:
            indexers = {}

        indexers.update(indexers_kwargs)

        if not indexers:
            raise ValueError("`indexers` cannot be empty. Provide either kwargs or `indexers`")

        self._indexers = indexers
        self._ignore_missing = ignore_missing
        self._tolerance = tolerance
        self._isel = isel

    # @property
    # def _info_(self):
    #     return dict(**self._indexers, ignore_missing=self._ignore_missing, tolerance=self._tolerance, isel=self._isel)

    def apply(self, dataset: xr.Dataset) -> xr.Dataset:
        for key, value in self._indexers.items():
            if self._ignore_missing and key not in dataset:
                continue

            # if isinstance(value, (tuple, list)): #Apparently .sel with list is real slow, attempting around that
            #     return xr.concat([select(**{key: i}, tolerance =tolerance, ignore_missing = ignore_missing)(dataset) for i in value], dim = key)

            try:
                if not self._isel:
                    dataset = dataset.sel(
                        **{key: value},
                        method="nearest" if self._tolerance is not None else None,
                        tolerance=self._tolerance,
                    )
                else:
                    dataset = dataset.isel(
                        **{key: value},
                    )
            except KeyError as e:
                raise DataNotFoundError(f"Selecting data with {key}: {value} raised an error") from e

        return dataset


@BackwardsCompatibility(Select)
def select(*args, **kwargs) -> Transform: ...


class Drop(Transform):
    """Drop items from Dataset"""

    def __init__(
        self,
        coordinates: list[Hashable] | tuple[Hashable] | Hashable | None = None,
        *extra_coords: Hashable,
        ignore_missing: bool = False,
    ):
        """
        Drop Items from xr.Dataset

        Args:
            coordinates (list[Hashable] | tuple[Hashable] | Hashable | None):
                Coordinates to drop. Defaults to None.
            ignore_missing (bool, optional):
                Ignore coordinates not in dataset. Defaults to False
        Returns:
            (Transform):
                Transform to apply drop
        """
        super().__init__()
        self.record_initialisation()

        if coordinates is None:
            coordinates = []

        coordinates = coordinates if isinstance(coordinates, (list, tuple)) else [coordinates]
        coordinates = [*coordinates, *extra_coords]

        self._coordinates = coordinates
        self._ignore_missing = ignore_missing

    # @property
    # def _info_(self):
    #     return dict(coordinates=self._coordinates, ignore_missing=self._ignore_missing)

    def apply(self, dataset: xr.Dataset) -> xr.Dataset:
        for i in self._coordinates:
            if self._ignore_missing and i not in dataset.coords:
                continue
            dataset = dataset.drop(i)
        return dataset


@BackwardsCompatibility(Drop)
def drop(*args, **kwargs) -> Transform: ...


def cast_to_int(value):
    try:
        if int(value) == value:
            value = int(value)
    except Exception:
        pass
    return value


class Flatten(Transform):
    """Flatten a coordinate in a dataset into seperate variables"""

    def __init__(
        self, coordinate: Hashable | list[Hashable] | tuple[Hashable], *extra_coordinates, skip_missing: bool = False
    ):
        """
        Flatten a coordinate in a dataset with each point being made a seperate data var

        Args:
            coordinate (Hashable | list[Hashable] | tuple[Hashable] | None):
                Coordinates to flatten, either str or list of candidates.
            *extra_coordinates (optional):
                Arguments form of `coordinate`.
            skip_missing (bool, optional):
                Whether to skip data without the dims. Defaults to False

        Raises:
            ValueError:
                If invalid number of coordinates found
        """
        super().__init__()
        self.record_initialisation()

        coordinate = coordinate if isinstance(coordinate, (list, tuple)) else [coordinate]
        coordinate = [*coordinate, *extra_coordinates]

        self._coordinate = coordinate
        self._skip_missing = skip_missing

    # @property
    # def _info_(self):
    #     return dict(coordinate=self._coordinate, skip_missing=self._skip_missing)

    def apply(self, dataset: xr.Dataset) -> xr.Dataset:
        discovered_coord = list(set(self._coordinate).intersection(set(dataset.coords)))

        if len(discovered_coord) == 0:
            if self._skip_missing:
                return dataset

            raise ValueError(
                f"{self._coordinate} could not be found in dataset with coordinates {list(dataset.coords)}.\n"
                "Set 'skip_missing' to True to skip this."
            )

        elif len(discovered_coord) > 1:
            transforms = TransformCollection(*[flatten(coord) for coord in discovered_coord])
            return transforms(dataset)

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
                coord_val = cast_to_int(dataset[var][discovered_coord].values)
                new_ds[f"{var}{coord_val}"] = Drop(discovered_coord, ignore_missing=True)(dataset[var])

            else:
                for coord_val in dataset[discovered_coord]:
                    coord_val = cast_to_int(coord_val.values.item())

                    selected = dataset[var].sel(**{discovered_coord: coord_val})  # type: ignore
                    selected = selected.drop(discovered_coord)  # type: ignore
                    selected.attrs.update(**{discovered_coord: coord_val})

                    new_ds[f"{var}{coord_val}"] = selected
        return new_ds


@BackwardsCompatibility(Flatten)
def flatten(*args, **kwargs) -> Transform: ...


class Expand(Transform):
    """Inverse operation to `Flatten`"""

    def __init__(self, coordinate: Hashable | list[Hashable] | tuple[Hashable], *extra_coordinates):
        """
        Inverse operation to [flatten][pyearthtools.data.transforms.coordinate.flatten]

        Will find flattened variables and regroup them upon the extra coordinate

        Args:
            coordinate (Hashable | list[Hashable] | tuple[Hashable]):
                Coordinate to unflatten.
            *extra_coordinates (optional):
                Argument form of `coordinate`.
        """
        super().__init__()
        self.record_initialisation()

        if not isinstance(coordinate, (list, tuple)):
            coordinate = (coordinate,)

        coordinate = (*coordinate, *extra_coordinates)
        self._coordinate = coordinate

    # @property
    # def _info_(self):
    #     return dict(coordinate=self._coordinate)

    def apply(self, dataset: xr.Dataset) -> xr.Dataset | xr.DataArray:
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
            dataset = pyearthtools.data.transforms.attributes.SetType(**{str(coord): dtype})(dataset)

            ## Add stored encoding if there
            if f"{coord}-dtype" in dataset.attrs:
                dtype = dataset.attrs.pop(f"{coord}-dtype")
                dataset[coord].encoding.update(dtype=dtype)

        return dataset


@BackwardsCompatibility(Expand)
def expand(*args, **kwargs) -> Transform: ...


def SelectFlatten(
    coordinates: dict[str, tuple[Any] | Any] | None = None,
    tolerance: float = 0.01,
    **extra_coordinates,
) -> TransformCollection:
    """
    Select upon coordinates, and flatten said coordinate

    Args:
        coordinates (dict[str, tuple[Any] | Any] | None, optional):
            Coordinates and values to select.
            Must be coordinate in data Defaults to None.
        tolerance (float, optional):
            tolerance of selection. Defaults to 0.01.

    Returns:
        (TransformCollection):
            TransformCollection to select and Flatten
    """

    if coordinates is None:
        coordinates = {}
    coordinates.update(extra_coordinates)

    select_trans = select(coordinates, ignore_missing=True, tolerance=tolerance)
    flatten_trans = flatten(list(coordinates.keys()))

    return select_trans + flatten_trans


@BackwardsCompatibility(SelectFlatten)
def select_flatten(*args, **kwargs) -> TransformCollection: ...


class Assign(Transform):
    """Assign coordinates to object"""

    def __init__(self, coordinates: dict[str, Any] | None = None, as_dataarray: bool = False, **coordinate_kwargs):
        """
        Assign coordinates to Xarray Object.

        Uses `.assign_coords`

        Args:
            coordinates (dict[str, Any] | None, optional):
                Coordinates to assign. Defaults to None.
            as_dataarray (bool, optional):
                Assign coordinates seperately to each variable. Defaults to False.

        """
        super().__init__()
        self.record_initialisation()

        if coordinates is None:
            coordinates = {}
        coordinates = dict(coordinates)

        coordinates.update(dict(coordinate_kwargs))

        for key, val in coordinates.items():
            if isinstance(val, xr.DataArray):
                coordinates[key] = list(map(float, val.values))

        if len(coordinates.keys()) == 0:
            raise ValueError("Either `coordinates` or `kwargs` must be given.")

        self._coordinates = coordinates
        self._as_dataarray = as_dataarray

    def apply(self, dataset: xr.Dataset) -> xr.Dataset:
        if self._as_dataarray:
            for var in dataset.data_vars:
                dataset[var] = dataset[var].assign_coords(**self._coordinates)
            return dataset
        return dataset.assign_coords(**self._coordinates)

    # @property
    # def _info_(self) -> dict:
    #     return dict(as_dataarray=self._as_dataarray, **self._coordinates)


@BackwardsCompatibility(Assign)
def assign(*args, **kwargs) -> Transform: ...


class Pad(Transform):
    """
    Pad data
    """

    def __init__(self, coordinates: dict[str, Any] | None = None, **kwargs):
        """
        Pad data

        This will automatically pad the coordinate values with an odd reflection to allow periodicy.

        Args:
            coordinates (dict[str, Any] | None, optional):
                Coordinate pad_width. Defaults to None.
                From xarray docs.
                    Mapping with the form of {dim: (pad_before, pad_after)} describing the number of values
                    padded along each dimension. {dim: pad} is a shortcut for pad_before = pad_after = pad
            **kwargs (Any, optional):
                Any kwargs to pass to `.pad`
        """
        super().__init__()
        self.record_initialisation()

        if coordinates is None:
            coordinates = {}
        coordinates = dict(coordinates)

        self._coordinates = coordinates
        self._kwargs = kwargs

    def apply(self, dataset: xr.Dataset) -> xr.Dataset:
        padded_dataset = dataset.pad(self._coordinates)

        padded_dataset = padded_dataset.assign_coords(
            {
                coord: dataset[coord].pad({coord: self._coordinates[coord]}, mode="reflect", reflect_type="odd")
                for coord in self._coordinates.keys()
            }
        )
        return padded_dataset

    # @property
    # def _info_(self) -> Any | dict:
    #     return dict(coordinates=self._coordinates, **self._kwargs)


@BackwardsCompatibility(Pad)
def pad(*args, **kwargs) -> Transform: ...
