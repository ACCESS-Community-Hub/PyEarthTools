# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

from __future__ import annotations

import warnings
from typing import Any, Hashable, Literal, Iterable
import logging

import xarray as xr
import numpy as np

DASK_IMPORTED = False
try:
    import dask

    DASK_IMPORTED = True
except ImportError:
    DASK_IMPORTED = False

import edit.data

from edit.data.transform.transform import Transform, TransformCollection
from edit.data.warnings import EDITDataWarning
from edit.data.exceptions import DataNotFoundError

LOG = logging.getLogger(__name__)

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

    def _return(coord_orientation: VALID_COORDINATE_DEFINITIONS) -> VALID_COORDINATE_DEFINITIONS | Transform:
        if not transform:
            return coord_orientation
        return standard_longitude(coord_orientation)

    if any(data.longitude.values > 180):
        return _return("0-360")
    elif any(data.longitude < 0):
        return _return("-180-180")

    raise ValueError(f"Could not identify longitude coordinate from data. {data.longitude}")
    LOG.debug(f"Could not identify longitude coordinate from data. {data.longitude}")


def standard_longitude(type: VALID_COORDINATE_DEFINITIONS = "-180-180") -> Transform:
    """
    Standardise format of longitude.

    Shifts the longitude coordinate to that of the specified. Must be in ["-180-180", "0-360"]

    Args:
        type (VALID_COORDINATE_DEFINITIONS): Longitude Specification. Defaults to "-180-180".

    Returns:
        (Transform):
            Transform to apply standardisation
    """
    valid_types = ["-180-180", "0-360"]
    if type not in valid_types:
        raise KeyError(f"Invalid `type` passed, must be one of {valid_types} not {type}")

    class StandardLongitude0360(Transform):
        """Force Longitude to be between 0 & 360"""

        @property
        def _info_(self):
            return dict(type=type)

        def _standardise(self, dataset):
            func = lambda x: x % 360
            dataset = dataset.assign_coords(longitude=func(dataset.longitude))
            return dataset.sortby("longitude")

        def apply(self, dataset: xr.Dataset) -> xr.Dataset:
            if "longitude" in dataset.coords and (dataset.longitude < 0).any():
                if DASK_IMPORTED:
                    with dask.config.set(**{"array.slicing.split_large_chunks": True}):
                        dataset = self._standardise(dataset)
                else:
                    dataset = self._standardise(dataset)
            elif "longitude" not in dataset.coords:
                warnings.warn(
                    f"Could not move longitude to 0-360, either 'longitude' is not in coords, or none lower than 0.",
                    EDITDataWarning,
                )
            return dataset

    class StandardLongitude180180(Transform):
        """Force Longitude to be between -180 & 180"""

        @property
        def _info_(self):
            return dict(type=type)

        def _standardise(self, dataset):
            func = lambda x: ((x + 180) % 360) - 180
            # (180 - abs(x - 180)) * np.sign((x - 180)) * -1

            dataset = dataset.assign_coords(longitude=func(dataset.longitude))
            return dataset.sortby("longitude")

        def apply(self, dataset: xr.Dataset) -> xr.Dataset:
            if "longitude" in dataset.coords and (dataset.longitude > 180).any():
                if DASK_IMPORTED:
                    with dask.config.set(**{"array.slicing.split_large_chunks": True}):
                        dataset = self._standardise(dataset)
                else:
                    dataset = self._standardise(dataset)
            elif "longitude" not in dataset.coords:
                warnings.warn(
                    f"Could not move longitude to -180-180, either 'longitude' is not in coords, or none greater than 180.",
                    EDITDataWarning,
                )
            return dataset

    return StandardLongitude180180() if type == "-180-180" else StandardLongitude0360()


def reindex(
    coordinates: dict[str, Literal["reversed", "sorted"] | Iterable] | xr.Coordinates | None = None, **coords
) -> Transform:
    """
    Reindex coordinates

    Can be sorted, or in set list

    Args:
        coordinates (dict[str, Literal['reversed','sorted'] | Iterable | xr.Coordinates] | None, optional):
            Coordinate to reindex, and Iterable to reindex at.
            If 'reversed' or 'sorted', take current coord and sort.
            If `xr.Coordinates`, use any coordinates with len > 1.
            Defaults to None.


    Returns:
        (Transform):
            Reindex transforrm
    """
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

    class ReIndex(Transform):
        """Reindex coordinates"""

        @property
        def _info_(self):
            return dict(**coordinates)

        def apply(self, dataset: xr.Dataset):
            for coord, index_op in coordinates.items():
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

    return ReIndex()


def force_standard_coordinate_names(replacement_dictionary: dict | None = None, **repl_kwargs) -> Transform:
    """
    Convert xr.Dataset Coordinate Names into Standard Naming Scheme

    Args:
        replacement_dictionary (dict | None, optional):
            Dictionary assigning name replacements [old: new].
            One of replacement_dictionary or repl_kwargs must be provided. Defaults to None.
        **repl_kwargs (dict, optional):
            Kwarg version of replacement_dictionary

    Returns:
        (Transform):
            Transform to convert names
    """
    if replacement_dictionary is None:
        replacement_dictionary = {}

    replacement_dictionary.update(repl_kwargs)

    class ConformNaming(Transform):
        """Force Standard Dimension Names"""

        @property
        def _info_(self):
            return dict(**replacement_dictionary)

        def apply(self, dataset: xr.Dataset):
            for correctname, falsenames in replacement_dictionary.items():
                for falsename in set(falsenames) & set(dataset.dims):
                    dataset = dataset.rename({falsename: correctname})

                for falsename in set(falsenames) & set(dataset.coords):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")  # TODO  UserWarning Raised below
                        dataset = dataset.rename({falsename: correctname})
            return dataset

    return ConformNaming()


def select(
    indexers: dict[str, Any] | None = None,
    *,
    ignore_missing: bool = False,
    tolerance: float | None = None,
    isel: bool = False,
    **indexers_kwargs,
) -> Transform:
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

    Returns:
        (Transform):
            Transform to apply selection
    """
    if indexers is None:
        indexers = {}

    indexers.update(indexers_kwargs)

    if not indexers:
        raise ValueError("`indexers` cannot be empty. Provide either kwargs or `indexers`")

    class Select(Transform):
        """Select on coordinates"""

        @property
        def _info_(self):
            return dict(**indexers, ignore_missing=ignore_missing, tolerance=tolerance)

        def apply(self, dataset: xr.Dataset) -> xr.Dataset:
            for key, value in indexers.items():
                if ignore_missing and key not in dataset:
                    continue

                # if isinstance(value, (tuple, list)): #Apparently .sel with list is real slow, attempting around that
                #     return xr.concat([select(**{key: i}, tolerance =tolerance, ignore_missing = ignore_missing)(dataset) for i in value], dim = key)

                try:
                    if not isel:
                        dataset = dataset.sel(
                            **{key: value},
                            method="nearest" if tolerance is not None else None,
                            tolerance=tolerance,
                        )
                    else:
                        dataset = dataset.isel(
                            **{key: value},
                        )
                except KeyError as e:
                    raise DataNotFoundError(f"Selecting data with {key}: {value} raised an error") from e
            return dataset

    return Select()


def drop(
    coordinates: list[Hashable] | tuple[Hashable] | Hashable | None = None,
    *extra_coords: Hashable,
    ignore_missing: bool = False,
) -> Transform:
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
    if coordinates is None:
        coordinates = []

    coordinates = coordinates if isinstance(coordinates, (list, tuple)) else [coordinates]
    coordinates = [*coordinates, *extra_coords]

    class Drop(Transform):
        """Drop coordinates from dataset"""

        @property
        def _info_(self):
            return dict(coordinates=coordinates, ignore_missing=ignore_missing)

        def apply(self, dataset: xr.Dataset) -> xr.Dataset:
            for i in coordinates:
                if ignore_missing and i not in dataset.coords:
                    continue
                dataset = dataset.drop(i)
            return dataset

    return Drop()


def cast_to_int(value):
    try:
        if int(value) == value:
            value = int(value)
    except Exception:
        pass
    return value


def flatten(
    coordinate: Hashable | list[Hashable] | tuple[Hashable], *extra_coordinates, skip_missing: bool = False
) -> Transform:
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

    Returns:
        (Transform):
            Transform to apply flatten
    """

    coordinate = coordinate if isinstance(coordinate, (list, tuple)) else [coordinate]
    coordinate = [*coordinate, *extra_coordinates]

    class Flatten(Transform):
        """
        Flatten coordinate dim of dataset, converting into seperate variables
        """

        @property
        def _info_(self):
            return dict(coordinate=coordinate, skip_missing=skip_missing)

        def apply(self, dataset: xr.Dataset) -> xr.Dataset:
            discovered_coord = list(set(coordinate).intersection(set(dataset.coords)))

            if len(discovered_coord) == 0:
                if skip_missing:
                    return dataset

                raise ValueError(
                    f"{coordinate} could not be found in dataset with coordinates {dataset.coords}.\n"
                    "Set 'skip_missing' to True to skip this."
                )

            elif len(discovered_coord) > 1:
                transforms = TransformCollection(*[flatten(coord) for coord in discovered_coord])
                return transforms(dataset)

            discovered_coord = discovered_coord[0]

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

                if coord_size.size == 1:
                    coord_val = cast_to_int(dataset[var][discovered_coord].values)
                    new_ds[f"{var}{coord_val}"] = drop(discovered_coord, ignore_missing=True)(dataset[var])

                else:
                    for coord_val in dataset[discovered_coord]:
                        coord_val = cast_to_int(coord_val.values)

                        selected = dataset[var].sel(**{discovered_coord: coord_val})
                        selected = selected.drop(discovered_coord)
                        selected.attrs.update(**{discovered_coord: coord_val})

                        new_ds[f"{var}{coord_val}"] = selected
            return new_ds

    return Flatten()


def expand(coordinate: Hashable | list[Hashable] | tuple[Hashable], *extra_coordinates) -> Transform:
    """Inverse operation to [flatten][edit.data.transform.coordinate.flatten]

    Will find flattened variables and regroup them upon the extra coordinate

    Args:
        coordinate (Hashable | list[Hashable] | tuple[Hashable]):
            Coordinate to unflatten.
        *extra_coordinates (optional):
            Argument form of `coordinate`.

    Returns:
        (TransformCollection):
            TransformCollection to expand dataset
    """

    if not isinstance(coordinate, (list, tuple)):
        coordinate = (coordinate,)

    coordinate = (*coordinate, *extra_coordinates)

    class Expand(Transform):
        """
        Expand flattened dimensions in dataset
        """

        @property
        def _info_(self):
            return dict(coordinate=coordinate)

        def apply(self, dataset: xr.Dataset) -> xr.Dataset:
            dataset = dataset.__class__(dataset)

            for coord in coordinate:
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

                dataset = xr.combine_by_coords(components)
                dataset = edit.data.transform.attributes.set_type(**{coord: "int32"})(dataset)

                ## Add stored encoding if there
                if f"{coord}-dtype" in dataset.attrs:
                    dtype = dataset.attrs.pop(f"{coord}-dtype")
                    dataset[coord].encoding.update(dtype=dtype)

            return dataset

    return Expand()


def select_flatten(
    coordinates: dict[str, tuple[Any] | Any] | None = None, tolerance: float = 0.01, **extra_coordinates
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


def assign(coordinates: dict[str, Any] | None = None, as_dataarray: bool = False, **coordinate_kwargs) -> Transform:
    """
    Assign coordinates to Xarray Object.

    Uses `.assign_coords`

    Args:
        coordinates (dict[str, Any] | None, optional):
            Coordinates to assign. Defaults to None.
        as_dataarray (bool, optional):
            Assign coordinates seperately to each variable. Defaults to False.

    Returns:
        (Transform):
            Transform to assign coordinates
    """
    if coordinates is None:
        coordinates = {}
    coordinates = dict(coordinates)

    coordinates.update(dict(coordinate_kwargs))

    for key, val in coordinates.items():
        if isinstance(val, xr.DataArray):
            coordinates[key] = list(map(float, val.values))

    if len(coordinates.keys()) == 0:
        raise ValueError("Either `coordinates` or `kwargs` must be given.")

    class AssignCoords(Transform):
        """
        Assign coordinates to xr.Dataset
        """

        def apply(self, dataset: xr.Dataset) -> xr.Dataset:
            if as_dataarray:
                for var in dataset.data_vars:
                    dataset[var] = dataset[var].assign_coords(**coordinates)
                return dataset
            return dataset.assign_coords(**coordinates)

        @property
        def _info_(self) -> dict:
            return dict(as_dataarray=as_dataarray, **coordinates)

    return AssignCoords()


# def ensure_time(time):
#     class EnsureTime(Transform):
#         """
#         Force data to have valid time coordinate
#         """
#         def apply(self, dataset: xr.Dataset) -> xr.Dataset:
#             return super().apply(dataset)


def pad(coordinates: dict[str, Any] | None = None, **kwargs) -> Transform:
    """
    Create a transform to pad data.

    This will automatically pad the coordinate values with an odd reflection to allow periodicy.

    Args:
        coordinates (dict[str, Any] | None, optional):
            Coordinate pad_width. Defaults to None.
            From xarray docs.
                Mapping with the form of {dim: (pad_before, pad_after)} describing the number of values
                padded along each dimension. {dim: pad} is a shortcut for pad_before = pad_after = pad
        **kwargs (Any, optional):
            Any kwargs to pass to `.pad`

    Returns:
        (Transform):
            Transform to pad coordinates.
    """
    if coordinates is None:
        coordinates = {}
    coordinates = dict(coordinates)

    class PadCoords(Transform):
        """
        Pad coordinates on Dataset
        """

        def apply(self, dataset: xr.Dataset) -> xr.Dataset:
            padded_dataset = dataset.pad(coordinates, **kwargs)
            padded_dataset = padded_dataset.assign_coords(
                {
                    coord: dataset[coord].pad({coord: coordinates[coord]}, mode="reflect", reflect_type="odd")
                    for coord in coordinates.keys()
                }
            )
            return padded_dataset

        @property
        def _info_(self) -> Any | dict:
            return dict(coordinates=coordinates, **kwargs)

    return PadCoords()


__all__ = [
    "pad",
    "assign",
    "select_flatten",
    "expand",
    "flatten",
    "drop",
    "select",
    "force_standard_coordinate_names",
    "reindex",
    "standard_longitude",
    "get_longitude",
]
