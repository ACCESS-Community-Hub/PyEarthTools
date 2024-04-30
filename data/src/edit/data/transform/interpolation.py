# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

from __future__ import annotations

import xarray as xr
from xarray.core.types import InterpOptions
import numpy as np

xESMF_IMPORTED = True
try:
    import xesmf as xe
except ImportError:
    xESMF_IMPORTED = False


import edit.data
from edit.data.transform.transform import Transform
from edit.data.transform.utils import parse_dataset


class InterpolateTransform:
    """Base Interpolation Transforms"""

    def __new__(
        cls, reference_dataset: xr.Dataset | None = None, method: InterpOptions = "linear", **kwargs
    ) -> Transform:
        """
        Create Interpolation Transform

        Args:
            reference_dataset (xr.Dataset, optional):
                Reference Dataset to interpolate to. Defaults to None.
            method (InterpOptions, optional):
                Interpolation method. Defaults to "linear".
            **kwargs (Any):
                Extra kwargs to provide to xarray.interp

        Returns:
            Transform: Transform to interpolate datasets
        """
        if reference_dataset:
            return cls.like(reference_dataset, method=method, **kwargs)
        return cls.interp(method=method, **kwargs)

    @classmethod
    def interp(
        cls,
        method: InterpOptions = "linear",
        keep_encoding: bool = False,
        skip_missing: bool = False,
        pad: bool | int = False,
        **kwargs,
    ) -> Transform:
        """
        Interpolation Transform passing kwargs

        Args:
            **kwargs (Any):
                Kwargs to pass to `xr.interp`. Should be variables with new coordinates to interpolate to.
                e.g.
                    latitude = [-90,-80,...,80,90]
            method (InterpOptions, optional):
                Method to use for interpolate. Defaults to "linear".
                Must be one of xarray.interp methods

                "linear", "nearest", "zero", "slinear", "quadratic", "cubic", "polynomial", "barycentric", "krog", "pchip", "spline", "akima"
            keep_encoding (bool, optional):
                Whether to keep the encoding of the incoming dataset. Defaults to False.
            pad (bool | int, optional):
                Whether to pad all coords by 1. If `int` size to pad by.
                Defaults to False.


        Returns:
            Transform: Transform to interpolate datasets
        """

        class InterpTransform(Transform):
            """Interpolate given Dataset with specified arguments"""

            @property
            def _info_(self):
                ref_kwargs = dict(kwargs)
                for key, value in ref_kwargs.items():
                    if isinstance(value, xr.DataArray):
                        value = [x for x in value.values]
                    if isinstance(value, list):
                        try:
                            value = [float(x) for x in value]
                        except TypeError:
                            pass

                    ref_kwargs[key] = value
                return dict(
                    **ref_kwargs, method=method, keep_encoding=keep_encoding, skip_missing=skip_missing, pad=pad
                )

            def apply(self, dataset: xr.Dataset) -> xr.Dataset:
                if keep_encoding:
                    encod = edit.data.transform.attributes.set_encoding(reference=dataset)
                else:
                    encod = lambda x: x
                _kwargs = dict(kwargs)
                if skip_missing:
                    _kwargs = {key: _kwargs[key] for key in set(_kwargs.keys()).intersection(dataset.coords)}

                if pad:
                    dataset = edit.data.transform.coordinates.pad({k: int(pad) for k in _kwargs})(dataset)
                return encod(dataset.interp(**kwargs, method=method))

        return InterpTransform()

    @classmethod
    def like(
        cls,
        reference_dataset: xr.Dataset | str,
        method: InterpOptions = "linear",
        drop_coords: str | list[str] | None = None,
        pad: bool | int = False,
        **kwargs,
    ) -> Transform:
        """
        From reference dataset setup interpolation transform

        Args:
            reference_dataset (xr.Dataset | str):
                Dataset to use to set coords. Can be path to dataset to open
            method (InterpOptions, optional):
                Method to use in interpolation. Defaults to "linear".
            drop_coords (str | list[str], optional):
                Coords to drop from reference dataset. Defaults to None.
            pad (bool | int, optional):
                Whether to pad all coords by 1. If `int` size to pad by.
                Defaults to False.
        Returns:
            (Transform):
                Transform to interpolate dataset like reference_dataset
        """
        reference_dataset = parse_dataset(reference_dataset)
        if not isinstance(reference_dataset, (xr.Dataset, xr.DataArray)):
            raise TypeError(f"Cannot interpolate like {type(reference_dataset)}: {reference_dataset}.")

        if drop_coords:
            if isinstance(drop_coords, str):
                drop_coords = [drop_coords]
            for coord in drop_coords:
                if coord in reference_dataset.coords:
                    reference_dataset = reference_dataset.drop_vars(coord)

        coords = dict(reference_dataset.coords)

        return cls.interp(method=method, **kwargs, **coords, pad=pad)  # type: ignore

    @classmethod
    def xesmf(cls, reference_dataset: xr.Dataset | None = None, method: str = "bilinear", **coords) -> Transform:
        """Create Transform using xesmf

        Either `reference_dataset` or `coords` must be given

        Args:
            reference_dataset (xr.Dataset, optional):
                Reference Dataset. Defaults to None.
            **coords (tuple):
                Coordinates to create reference_dataset from.
                Can be fully created or tuple to use to fill np.arange
                Either:
                    lat = (["lat"], np.arange(16, 75, 1.0))
                or
                    lat = (16, 75, 1.0)
            method (str, optional):
                Method to use. Defaults to "bilinear".

        Raises:
            ImportError:
                xesmf could not be imported
            KeyError:
                No arguments given

        Returns:
            (Transform):
                Transform to interpolate dataset
        """
        if not xESMF_IMPORTED:
            raise ImportError(f"xesmf could not be imported")
        if not reference_dataset and not coords:
            raise KeyError(f"Either 'reference_dataset' or 'coords' must be given")

        def get_reference(reference_dataset: xr.Dataset = None, coords: dict = None):
            if reference_dataset:
                return reference_dataset

            try:
                return xr.Dataset(coords)
            except ValueError as e:
                pass

            new_coords = {}
            for key, value in coords:
                new_coords[key] = ([key], np.arange(*value))
            return xr.Dataset(coords)

        ds_out = get_reference(reference_dataset, coords)

        class xESMFInterpTransform(Transform):
            """Interpolate given Dataset with specified arguments using xesmf"""

            @property
            def _info_(self):
                return dict(**coords, method=method)

            def apply(self, dataset: xr.Dataset) -> xr.Dataset:
                regridder = xe.Regridder(dataset, ds_out, "bilinear")

                return regridder(dataset)

        return xESMFInterpTransform()


def interpolate_na(
    dim: str,
    method: InterpOptions = "linear",
    keep_encoding: bool = False,
    fill_value: str | None = "extrapolate",
    **kwargs,
) -> Transform:
    """
    Interpolate Nan Transform.

    Uses `xarray.ds.interpolate_na`, see for all kwargs.

    Automatically reindexes to be monotonic, and reverts before pass back.

    Args:
        **kwargs (Any):
            Kwargs to pass to `xr.interpolate_na`

        method (InterpOptions, optional):
            Method to use for interpolate. Defaults to "nearest".
            Must be one of xarray.interp methods

            "linear", "nearest", "zero", "slinear", "quadratic", "cubic", "polynomial", "barycentric", "krog", "pchip", "spline", "akima"
        keep_encoding (bool, optional):
            Whether to keep the encoding of the incoming dataset. Defaults to False.
        fill_value (str | None, optional):
            See `scipy.interpolate.interp1d`.

    Returns:
        Transform: Transform to interpolate nan on datasets
    """

    class InterpNanTransform(Transform):
        """Interpolate Nans on Dataset"""

        @property
        def _info_(self):
            return dict(dim=dim, method=method, keep_encoding=keep_encoding, fill_value=fill_value)

        def apply(self, dataset: xr.Dataset) -> xr.Dataset:
            if keep_encoding:
                encod = edit.data.transform.attributes.set_encoding(reference=dataset)
            else:
                encod = lambda x: x

            revert_reindex = edit.data.transform.coordinates.reindex(dataset.coords)  # type: ignore
            reindex = edit.data.transform.coordinates.reindex({key: "sorted" for key in dataset.coords if len(np.atleast_1d(dataset.coords[key].values)) > 1})  # type: ignore

            if fill_value is not None:
                kwargs["fill_value"] = fill_value
            return revert_reindex(encod(reindex(dataset).interpolate_na(dim=dim, method=method, **kwargs)))

    return InterpNanTransform()
