# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

from __future__ import annotations

from pathlib import Path
from typing import Callable, Union

import numpy as np
import pandas as pd
import xarray as xr
import yaml

try:
    import geopandas as gpd

    GEOPANDAS_IMPORTED = True
except ImportError:
    GEOPANDAS_IMPORTED = False

from edit.data.transform import Transform
from edit.data.transform.utils import parse_dataset

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


class RegionTransform:
    """
    Functions to create a Transform to change a Dataset's geospatial extent
    """

    def __new__(cls, *args, **kwargs) -> Transform:
        """
        Automatically create a RegionTransform

        Args:
            *args (Any): Can be

                Four floats, specifying:
                    min_lat, max_lat, min_lon, max_lon

                Reference Dataset:
                    To get extent from

                Key
                    Lookup key

                Shapefile
                    Shapefile

        Returns:
            Transform: RegionTransform to cut a dataset's extent
        """
        first_kwarg = None if not kwargs else kwargs[list(kwargs.keys())[0]]

        if len(args) == 1:
            if isinstance(args[0], (xr.Dataset, xr.DataArray)) or isinstance(first_kwarg, (xr.Dataset, xr.DataArray)):
                return cls.like(args[0])
            elif isinstance(args[0], str) or isinstance(first_kwarg, str):
                return cls.lookup(args[0])
            elif isinstance(args[0], (tuple, list)) or isinstance(first_kwarg, (tuple, list)):
                return cls.bounding(*args[0], **kwargs)
            else:
                return cls.from_shapefile(args[0], **kwargs)
        try:
            return cls.bounding(*args, **kwargs)
        except Exception:
            return cls.from_geosearch(*args, **kwargs)

    @staticmethod
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

        return RegionTransform.bounding(min_lat, max_lat, min_lon, max_lon)

    @staticmethod
    def sel(**sel_kwargs):
        class SelectCut(Transform):
            """Cut Dataset with specified select kwargs"""

            @property
            def _info_(self):
                return dict(**sel_kwargs)

            def apply(self, dataset: xr.Dataset):
                # TODO Add automatic coordinate analysis, slice on 0-360 with a ds with -180-180
                subset_dataset = dataset.sel(**sel_kwargs)
                return subset_dataset

        return SelectCut()

    @staticmethod
    def isel(**sel_kwargs):
        class iSelectCut(Transform):
            """Cut Dataset with specified select kwargs"""

            @property
            def _info_(self):
                return dict(**sel_kwargs)

            def apply(self, dataset: xr.Dataset):
                # TODO Add automatic coordinate analysis, slice on 0-360 with a ds with -180-180
                subset_dataset = dataset.isel(**sel_kwargs)
                return subset_dataset

        return iSelectCut()

    @staticmethod
    def bounding(
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        *,
        name: str | None = None,
    ) -> Transform:
        """
        Use Bounding Coordinates to transform geospatial extent

        Args:
            min_lat (float): Minimum Latitude  to slice with
            max_lat (float): Maximum Latitude  to slice with
            min_lon (float): Minimum Longitude to slice with
            max_lon (float): Maximum Longitude to slice with

        Returns:
            Transform: Transform to cut region to given bounding box
        """

        min_lat, max_lat = order(min_lat, max_lat)
        min_lon, max_lon = order(min_lon, max_lon)

        doc = f"Cut Dataset to {name} region"

        class BoundingCut(Transform):
            """Cut Dataset to specified Bounding Box"""

            @property
            def _info_(self):
                return dict(
                    name=name,
                    min_lat=min_lat,
                    max_lat=max_lat,
                    min_lon=min_lon,
                    max_lon=max_lon,
                )

            def apply(self, dataset: xr.Dataset):
                # TODO Add automatic coordinate analysis, slice on 0-360 with a ds with -180-180
                subset_dataset = dataset.sel(latitude=slice(min_lat, max_lat), longitude=slice(min_lon, max_lon))
                if check_shape(subset_dataset) == 0:
                    subset_dataset = dataset.sel(
                        latitude=slice(max_lat, min_lat),
                        longitude=slice(min_lon, max_lon),
                    )
                return subset_dataset

        return BoundingCut(docstring=doc if name else None)

    @staticmethod
    def point_box(point: tuple[float], size: float, name: str | None = None) -> Transform:
        """
        Create a region bounding box of `size` around `point`

        Args:
            point (tuple[float]):
                Latitude and Longitude point
            size (float):
                Size in degrees to expand the box
                Total box width / length = `size` * 2
            name (str | None, optional):
                Name to pass through. Defaults to None.

        Returns:
            (Transform):
                Transform to cut region to bounding box around point
        """
        edges = (tuple(map(lambda x: x - size, point)), tuple(map(lambda x: x + size, point)))
        return RegionTransform.bounding(edges[0][0], edges[1][0], edges[0][1], edges[1][1], name=name)

    @staticmethod
    def lookup(key: str, regionfile: str | Path = RegionLookupFILE) -> Transform:
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
        lookup_dict = RegionTransform.lookup_dict(regionfile)

        if key not in lookup_dict:
            raise KeyError(f"{key} not in {RegionLookupFILE.stem}. Must be one of {list(lookup_dict.keys())}")

        bounding_box = lookup_dict[key]
        if isinstance(bounding_box, dict):
            return RegionTransform.bounding(**bounding_box, name=key)

        return RegionTransform.bounding(*bounding_box, name=key)

    @staticmethod
    def lookup_dict(regionfile: str | Path = RegionLookupFILE) -> dict[str, tuple]:
        """Get Region Lookup Dictionary"""
        with open(regionfile) as file:
            lookup_dict = yaml.safe_load(file)
        return lookup_dict

    @staticmethod
    def from_shapefile(shapefile, crs: str | None = None, name: str | None = None) -> Transform:
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
            name (str | None, optional):
                Name of shapefile, for docstring reference. Defaults to None.

        Raises:
            ImportError:
                If geopandas cannot be imported

        Returns:
            (Transform):
                Transform to cut dataset to shapefile mask
        """

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

        class ShapeFileCut(Transform):
            def apply(self, dataset: xr.Dataset):
                import rioxarray

                dataset.rio.write_crs(crs, inplace=True)
                dataset = dataset.rio.clip(shapefile)
                if "crs" in dataset.coords:
                    dataset = dataset.drop_vars("crs")
                return dataset

            def plot(self, **kwargs):
                shapefile.plot(**kwargs)

            @property
            def _info_(self):
                return dict(crs=str(crs), name=name)

            @property
            def _doc_(self):
                doc = "Cut Dataset to shapefile"
                if name:
                    doc += f": {name}"
                return doc

        return ShapeFileCut()

    shapefile = from_shapefile

    @staticmethod
    def from_geosearch(
        key: str,
        column: str | None = None,
        value: list[str] | str | None = None,
        crs: str | None = None,
        **kwargs,
    ) -> Transform:
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
        Returns:
            (Transform):
                Transform to cut dataset to shapefile mask
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

        func = RegionTransform.from_shapefile(shapefile, crs=crs, name=str(value))
        # setattr(func, '__doc__', f"Cut Dataset to Shapefile, from {key}, {column}, {value}")
        return func

    geosearch = from_geosearch
