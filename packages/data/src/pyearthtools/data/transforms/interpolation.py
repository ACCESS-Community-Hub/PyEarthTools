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
from typing import Optional

import xarray as xr
from xarray.core.types import InterpOptions
import numpy as np


import pyearthtools.data
from pyearthtools.data.transforms.transform import Transform
from pyearthtools.data.transforms.utils import parse_dataset

from pyearthtools.utils.decorators import BackwardsCompatibility

xESMF_IMPORTED = True
try:
    import xesmf as xe  # type: ignore
except ImportError:
    xESMF_IMPORTED = False


class Interpolate(Transform):
    """Interpolation Transform"""

    def __init__(
        self,
        method: InterpOptions = "linear",
        keep_encoding: bool = False,
        skip_missing: bool = False,
        pad: bool | int = False,
        **kwargs,
    ):
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
            skip_missing (bool, optional):
                Skip missing dimensions as given in `kwargs` but not in dataset. Defaults to False.
            pad (bool | int, optional):
                Whether to pad all coords by 1. If `int` size to pad by.
                Defaults to False.


        Returns:
            Transform: Transform to interpolate datasets
        """
        super().__init__()
        self.record_initialisation()

        self._method = method
        self._keep_encoding = keep_encoding
        self._skip_missing = skip_missing
        self._pad = pad
        self._kwargs = kwargs

        self.update()

    def update(self):
        ref_kwargs = dict(self._kwargs)
        for key, value in ref_kwargs.items():
            if isinstance(value, xr.DataArray):
                value = [x for x in value.values]
            if isinstance(value, list):
                try:
                    value = [float(x) for x in value]
                except TypeError:
                    pass

            ref_kwargs[key] = value
        self.update_initialisation(ref_kwargs)

    def apply(self, dataset: xr.Dataset) -> xr.Dataset:
        if self._keep_encoding:
            encod = pyearthtools.data.transforms.attributes.set_encoding(reference=dataset)
        else:
            encod = lambda x: x  # noqa: E731
        _kwargs = dict(self._kwargs)
        if self._skip_missing:
            _kwargs = {key: _kwargs[key] for key in set(_kwargs.keys()).intersection(dataset.coords)}

        if self._pad:
            dataset = pyearthtools.data.transforms.coordinates.Pad({k: int(self._pad) for k in _kwargs})(dataset)
        return encod(dataset.interp(**self._kwargs, method=self._method))  # type: ignore


def like(
    reference_dataset: xr.Dataset | str,
    method: InterpOptions = "linear",
    drop_coords: str | list[str] | None = None,
    pad: bool | int = False,
    **kwargs,
):
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

    return Interpolate(method=method, **kwargs, **coords, pad=pad)  # type: ignore


class XESMF(Transform):
    """Interpolate using xesmf"""

    def __init__(
        self,
        reference_dataset: xr.Dataset | None = None,
        method: str = "bilinear",
        **coords,
    ):
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
        """
        super().__init__()
        self.record_initialisation()

        if not xESMF_IMPORTED:
            raise ImportError(f"xesmf could not be imported")
        if not reference_dataset and not coords:
            raise KeyError(f"Either 'reference_dataset' or 'coords' must be given")

        def get_reference(reference_dataset: Optional[xr.Dataset] = None, coords: Optional[dict] = None):
            if reference_dataset:
                return reference_dataset

            try:
                return xr.Dataset(coords)
            except ValueError as e:
                pass

            if coords:
                new_coords = {}
                for key, value in coords:
                    new_coords[key] = ([key], np.arange(*value))
                return xr.Dataset(coords)
            raise ValueError(f"Cannot parse interpolation config of {reference_dataset} and {coords}.")

        self._ds_out = get_reference(reference_dataset, coords)
        self._method = method

    def apply(self, dataset: xr.Dataset) -> xr.Dataset:
        regridder = xe.Regridder(dataset, self._ds_out, self._method)

        return regridder(dataset)


class InterpolateNan(Transform):
    """Interpolate Nan's"""

    def __init__(
        self,
        dim: str,
        method: InterpOptions = "linear",
        keep_encoding: bool = False,
        fill_value: str | None = "extrapolate",
        **kwargs,
    ):
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

        """
        super().__init__()
        self.record_initialisation()

        self._dim = dim
        self._method = method
        self._keep_encoding = keep_encoding
        self._fill_value = fill_value
        self._kwargs = kwargs

    def apply(self, dataset: xr.Dataset) -> xr.Dataset:
        if self._keep_encoding:
            encod = pyearthtools.data.transforms.attributes.set_encoding(reference=dataset)
        else:

            def encod(x):
                return x

        revert_reindex = pyearthtools.data.transforms.coordinates.reindex(dataset.coords)  # type: ignore
        reindex = pyearthtools.data.transforms.coordinates.reindex(
            {key: "sorted" for key in dataset.coords if len(np.atleast_1d(dataset.coords[key].values)) > 1}
        )  # type: ignore

        if self._fill_value is not None:
            self._kwargs["fill_value"] = self._fill_value
        return revert_reindex(
            encod(reindex(dataset).interpolate_na(dim=self._dim, method=self._method, **self._kwargs))
        )  # type: ignore


@BackwardsCompatibility(InterpolateNan)
def interpolate_na(*args, **kwargs): ...


### Model levels to pressure level transform needed


class ModelToPressureLevels(Transform):
    pass


# import iris
# import stratify
# import numpy as np

# def vertial_interpolation(air_pressure_mod_levels, specific_hum_mod_levels):
#     ''' Function to convert specific humidity on model levels into specific humidity on pressure levels. The desired pressure levels are hard coded into the function as they are static. We use the python package stratify to do the interpolation.

#     The interpolation will produce NaNs in certain places, for more information on this and why we can replace these with surface values, see notebooks/specific_hum_processing.ipynb
#     '''

#     pressure_in_hpa = np.flip(np.array([50,100,150,200,250,300,400,500,600,700,850,925,1000]))
#     pressure_in_pa = pressure_in_hpa * 100

#     # Do interpolation with stratify
#     specific_hum_pres_levels = stratify.interpolate(pressure_in_pa,
#                                             air_pressure_mod_levels.data,
#                                             specific_hum_mod_levels.data,
#                                             axis=0)

#     #extract the model level 0 to replace any NaNs with
#     mod_lev_constraint = iris.Constraint(model_level_number = 0)
#     sh_model_level_0_data = specific_hum_mod_levels.extract(mod_lev_constraint).data

#     for pressure_level in specific_hum_pres_levels:
#         pressure_level[np.isnan(pressure_level)] = sh_model_level_0_data[np.isnan(pressure_level)]

#     # reflip so we have the pressure levels in the same order as our other data
#     specific_hum_pres_levels = np.flip(specific_hum_pres_levels, axis = 0)

#     return specific_hum_pres_levels
