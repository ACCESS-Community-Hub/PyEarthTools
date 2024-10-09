# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Union

import numpy as np
import pandas as pd
import xarray as xr
import yaml

try:
    import geopandas as gpd

    GEOPANDAS_IMPORTED = True
except ImportError:
    GEOPANDAS_IMPORTED = False

from edit.data.transforms import Transform
from edit.data.transforms.utils import parse_dataset

from edit.utils.decorators import BackwardsCompatibility


RegionLookupFILE = Path(__file__).parent / "RegionLookup.yaml"


def check_shape(data: xr.Dataset | xr.DataArray) -> int:
    """
    Calculate multiplied shape of xarray data container

    Args:
        data (xr.Dataset | xr.DataArray): Data to find shape for

    Returns:
        int: Multiplied shape of data
    """
    if isinstance(data, xr.Dataset):
        return min([int(np.prod(data[var].shape)) for var in data])
    else:
        return int(np.prod(data.shape))


def order(*args):
    """Order arguments with sort & return as tuple"""
    args = list(args)
    args.sort()
    return tuple(args)


class Bounding(Transform):
    """Cut with Bounding box"""

    def __init__(self, min_lat: float, max_lat: float, min_lon: float, max_lon: float):
        """
        Use Bounding Coordinates to transform geospatial extent

        Args:
            min_lat (float): Minimum Latitude  to slice with
            max_lat (float): Maximum Latitude  to slice with
            min_lon (float): Minimum Longitude to slice with
            max_lon (float): Maximum Longitude to slice with
        """
        super().__init__()
        self.record_initialisation()

        self._min_lat, self._max_lat = order(min_lat, max_lat)
        self._min_lon, self._max_lon = order(min_lon, max_lon)

    def apply(self, dataset: xr.Dataset):
        # TODO Add automatic coordinate analysis, slice on 0-360 with a ds with -180-180
        subset_dataset = dataset.sel(
            latitude=slice(self._min_lat, self._max_lat), longitude=slice(self._min_lon, self._max_lon)
        )

        if check_shape(subset_dataset) == 0:
            subset_dataset = dataset.sel(
                latitude=slice(self._max_lat, self._min_lat),
                longitude=slice(self._min_lon, self._max_lon),
            )
        return subset_dataset


def like(dataset: xr.Dataset | xr.DataArray | str) -> Transform:
    """
    Use Reference Dataset to inform spatial extent
    & transform geospatial extent accordingly

    Args:
        dataset (xr.Dataset | str):
            Reference Dataset to use. Can be path to dataset to load

    Returns:
        (Transform): Transform to cut region to extent of given reference dataset
    """

    reference_dataset: xr.DataArray | xr.Dataset = parse_dataset(dataset)  # type: ignore

    min_lat = float(reference_dataset.latitude.min().data)
    max_lat = float(reference_dataset.latitude.max().data)
    min_lon = float(reference_dataset.longitude.min().data)
    max_lon = float(reference_dataset.longitude.max().data)

    return Bounding(min_lat, max_lat, min_lon, max_lon)


class Select(Transform):
    """Select"""

    def __init__(self, sel_kwargs: dict[str, Any] | None = None, **kwargs):
        """
        Select on a dataset with `sel_kwargs`
        """
        super().__init__()
        self.record_initialisation()

        self._sel_kwargs = dict(sel_kwargs or {})
        self._sel_kwargs.update(kwargs)

    def apply(self, dataset: xr.Dataset):
        # TODO Add automatic coordinate analysis, slice on 0-360 with a ds with -180-180
        subset_dataset = dataset.sel(**self._sel_kwargs)
        return subset_dataset


@BackwardsCompatibility(Select)
def sel(*args, **kwargs): ...


class ISelect(Transform):
    """ISelect"""

    def __init__(self, sel_kwargs: dict[str, Any] | None = None, **kwargs):
        """
        Index select on a dataset with `sel_kwargs`
        """
        super().__init__()
        self.record_initialisation()

        self._sel_kwargs = dict(sel_kwargs or {})
        self._sel_kwargs.update(kwargs)

    def apply(self, dataset: xr.Dataset):
        # TODO Add automatic coordinate analysis, slice on 0-360 with a ds with -180-180
        subset_dataset = dataset.isel(**self._sel_kwargs)
        return subset_dataset


@BackwardsCompatibility(ISelect)
def isel(*args, **kwargs): ...


def PointBox(point: tuple[float], size: float) -> Transform:
    """
    Create a region bounding box of `size` around `point`

    Args:
        point (tuple[float]):
            Latitude and Longitude point
        size (float):
            Size in degrees to expand the box
            Total box width / length = `size` * 2

    Returns:
        (Transform):
            Transform to cut region to bounding box around point
    """

    edges = (
        tuple(map(lambda x: x - size, point)),
        tuple(map(lambda x: x + size, point)),
    )
    return Bounding(edges[0][0], edges[1][0], edges[0][1], edges[1][1])


@BackwardsCompatibility(PointBox)
def point_box(*args, **kwargs): ...


def Lookup(key: str, regionfile: str | Path = RegionLookupFILE) -> Transform:
    """
    Use string to retrieve preset lat and lon extent to transform geospatial extent

    Args:
        key (str):
            Lookup key within the preset file
        regionfile (str | Path):
            Yaml File to look for keys in. Defaults to RegionLookupFILE

    Raises:
        KeyError:
            If key not in preset file

    Returns:
        (Transform):
            Transform to cut region to define bounding box
    """
    with open(regionfile) as file:
        lookup_dict = yaml.safe_load(file)

    if key not in lookup_dict:
        raise KeyError(f"{key} not in {RegionLookupFILE.stem}. Must be one of {list(lookup_dict.keys())}")

    bounding_box = lookup_dict[key]
    if isinstance(bounding_box, dict):
        return Bounding(**bounding_box)

    return Bounding(*bounding_box)


@BackwardsCompatibility(Lookup)
def lookup(*args, **kwargs): ...


class ShapeFile(Transform):
    def __init__(self, shapefile, crs: str | None = None):
        """
        Use Shapefile to create region bounding.

        Args:
            shapefile (Any | str):
                Shapefile to use
            crs (str | None, optional):
                Coordinate Reference System (CRS) to apply to data.
                Will check if `shapefile` has crs information and attempt to use if not provided.
                Otherwise an error will be raised.

                Can be any code accepted by `geopandas`. See
                (here)[https://geopandas.org/en/stable/docs/user_guide/projections.html#coordinate-reference-systems]

                Defaults to None.

        Raises:
            ImportError:
                If geopandas cannot be imported

        """
        super().__init__()
        self.record_initialisation()

        if not GEOPANDAS_IMPORTED:
            raise ImportError(f"geopandas could not be imported")

        if isinstance(shapefile, (str, Path)):
            shapefile = gpd.read_file(shapefile)

        if hasattr(shapefile, "geometry"):
            shapefile = shapefile.geometry

        if hasattr(shapefile, "crs"):
            crs = crs or shapefile.crs

        if crs is None:
            raise TypeError(
                f"Coordinate Reference System (CRS) cannot be None. Could not automatically find from shapefile"
            )
        self._shapefile = shapefile
        self._crs = crs

    def apply(self, dataset: xr.Dataset):
        import rioxarray

        dataset.rio.write_crs(self._crs, inplace=True)
        dataset = dataset.rio.clip(self._shapefile)
        if "crs" in dataset.coords:
            dataset = dataset.drop_vars("crs")
        return dataset

    def plot(self, **kwargs):
        self._shapefile.plot(**kwargs)


@BackwardsCompatibility(ShapeFile)
def from_shapefile(*args, **kwargs): ...


def Geosearch(
    key: str,
    column: str | None = None,
    value: list[str] | str | None = None,
    crs: str | None = None,
    **kwargs,
):
    """
    Using [static.geographic][edit.data.static.geographic] retrieve a Shapefile.
    Allows selection of geopandas file, column and value to filter by

    If no column nor value provided, use all geometry in geopandas file

    Args:
        key (str):
            A [Geographic][edit.data.static.geographic] search key
        column (str | None, optional):
            Column in geopandas to search in. Defaults to None.
        value (list[str] | str, optional):
            Values to search for, can be list. Defaults to None.
        crs (str | None, optional):
            Coordinate Reference System (CRS) to apply to data.
            Will check if `shapefile` has crs information and attempt to use if not provided.
            Otherwise an error will be raised.

            Can be any code accepted by `geopandas`. See
            (here)[https://geopandas.org/en/stable/docs/user_guide/projections.html#coordinate-reference-systems]
    """
    from edit.data.static import geographic

    geo = geographic(**kwargs)(key)
    geo = geo[~geo.geometry.isna()]
    if column:
        if isinstance(value, list):
            shapefile = pd.concat([geo[geo[column] == val] for val in value]).geometry
        else:
            shapefile = geo[geo[column] == value].geometry
    else:
        shapefile = geo.geometry

    return ShapeFile(shapefile, crs=crs)


@BackwardsCompatibility(Geosearch)
def from_geosearch(*args, **kwargs): ...
