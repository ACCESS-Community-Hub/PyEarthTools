# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

"""
Tesselation Tool

Provides ways to split data into patches, and reform patches back to full data
"""
from __future__ import annotations

from typing import Union
import warnings

import numpy as np
import xarray as xr

from edit.utils.data.tesselator import _patching
from edit.utils.warnings import TesselatorWarning
from edit.utils.exceptions import TesselatorException


class Tesselator:
    """
    Data Tesselator.

    Used to split a numpy or xarray object into patches of a given size, and given stride.

    Provides methods to stitch the patches back together into the input object.
    """

    def __init__(
        self,
        kernel_size: int,
        stride: int = None,
        padding: str = "reflect",
        coord_template: xr.Dataset | xr.DataArray = None,
        out_name: str = "Reconstructed",
        ignore_difference: bool = False,
    ):
        """Create Tesselator

        Args:
            kernel_size (int):
                Size of each individual kernel
            stride (int, optional):
                Distance between kernels.If none, set to kernel_size. Defaults to None.
            padding (str, optional):
                Padding operation, either str or function. Must be one of np.pad modes.
                If padding is None, patches will not consist of the whole data, and issues will arise with stitching.
                Defaults to "reflect".
            coord_template (xr.Dataset | xr.DataArray, optional):
                Set coordinate template for stitch output. Defaults to None.
            out_name (str, optional):
                Name of dataArray outputted from stitch. Defaults to "Reconstructed".
            ignore_difference (bool, optional):
                Quiet warnings about differences in shapes when undoing
        """

        self.kernel_size = [kernel_size] if isinstance(kernel_size, int) else list(kernel_size)

        if len(self.kernel_size) == 1:
            self.kernel_size = self.kernel_size * 2

        self.stride = stride or kernel_size
        self.stride = [self.stride] if isinstance(self.stride, int) else list(self.stride)
        if len(self.stride) == 1:
            self.stride = self.stride * 2

        if (np.array(self.stride) > np.array(self.kernel_size)).any():
            warnings.warn(
                f"Stride is larger than kernel size. ({self.stride} > {self.kernel_size}). "
                "While patching will work, stitching with this configuration will not.",
                TesselatorWarning,
            )

        self.padding = padding
        self.ignore_difference = ignore_difference

        self._coords = None
        self._dims = None
        self._attrs = {}

        self.out_name = out_name

        if coord_template:
            self._set_coords(coord_template)

        self._initial_shape = None
        self._post_shape = None
        self._return_type = None
        self._layout = None

    def _find_shape(self, data):
        if isinstance(data, np.ndarray):
            shape = data.shape
        elif isinstance(data, xr.Dataset):
            shape = (
                len(list(data.data_vars)),
                *data[list(data.data_vars)[0]].shape,
            )
        elif isinstance(data, xr.DataArray):
            shape = data.shape
        else:
            raise TypeError(f"Unable to find shape of {data!r}")
        return shape

    def _set_coords(self, data: Union[xr.DataArray, xr.Dataset, np.ndarray]):
        """From data find shape.
        If data is an xr DataArray or Dataset, save coordinates and dims for stitching

        Args:
            data (Union[xr.DataArray, xr.Dataset,  np.ndarray]): Template data to get attributes from

        Raises:
            TesselatorException: If new shape doesn't match the expected shape
        """

        shape = self._find_shape(data)

        if self._initial_shape and not shape == self._initial_shape:
            raise TesselatorException(
                f"Initial shape was {self._initial_shape!r} which doesn't match incoming shape {shape!r}"
            )
        else:
            self._initial_shape = shape

        if isinstance(data, np.ndarray) or data is None:
            self._return_type = "numpy"
            return

        elif isinstance(data, (xr.Dataset, xr.DataArray)):
            self._attrs["global"] = data.attrs
            self._return_type = type(data)

            if isinstance(data, xr.Dataset):
                self._variables = list(data.data_vars)

                for var in self._variables:
                    self._attrs[var] = data[var].attrs

            self._coords = {}
            self._dims = [None] * (len(data.coords) + 1)

            use_shape = list(self._initial_shape)
            for coord in data.coords:
                size = len(data[coord])
                self._dims[use_shape.index(size)] = coord
                use_shape[use_shape.index(size)] = 1e10

            while None in self._dims:
                self._dims.remove(None)

            for dim in self._dims:
                self._coords[dim] = data[dim].values

            if isinstance(data, xr.Dataset):
                self._dims = ["Variables"] + self._dims
                self._coords["Variables"] = self._variables

    def _get_coords(self) -> tuple[list, dict]:
        """Retrieve coords and dims from self

        Raises:
            AttributeError: If no template has been provided

        Returns:
            tuple[list, dict]: Dims, Coords & Attributes
        """

        if self._coords is None:
            raise AttributeError("No template has been provided, unable to assign coordinates")

        return list(self._dims), dict(self._coords), self._attrs

    def patch(
        self,
        input_data: xr.DataArray | xr.Dataset | np.ndarray,
        data_format: str = None,
        **kwargs,
    ) -> np.ndarray:
        """Patch incoming data into patches as configured

        Args:
            input_data (xr.DataArray | xr.Dataset | np.ndarray): Data to Patch
            data_format (str, optional): Format of Data if not normal. Defaults to None.
            **kwargs (Any, optional): Extra keyword args to be passed to [make_patches][edit.utils.data.tesselator._patching.patches.make_patches]

        Returns:
            (np.ndarray): Patches of data, with the first dimension being the squashed patch dim
        """

        self._set_coords(input_data)

        patches, layout = _patching.patches.make_patches(
            input_data,
            self.kernel_size,
            self.stride,
            data_format=data_format,
            padding=self.padding,
            **kwargs,
        )

        self._layout = layout
        self._post_shape = patches.shape

        return patches

    def stitch(
        self,
        input_data: np.ndarray,
        data_format: str = "TCHW",
        override: dict = None,
        var_select: str = None,
        as_numpy: bool = False,
    ) -> np.ndarray | xr.Dataset | xr.DataArray:
        """Stitch back together patches of data generated by this `Tesselator`

        If original data was an [xarray][xarray] object, this will attempt to reconstruct it with dims & coords intact

        Args:
            input_data (np.ndarray): Patches to stitch together, must have come from this Tesselator
            data_format (str, optional): Format of Data after patch fim. Defaults to "TCHW".
            override (dict, optional): Override for coordinates for xarray. Defaults to None.
            var_select (str, optional): If only one variable is given back, use this to select it. Defaults to None.
            as_numpy (bool, optional): Whether to return only as numpy. Defaults to False.

        Raises:
            NotImplementedError: If offset to remove padding is negative

        Returns:
            (np.ndarray | xr.Dataset | xr.DataArray): Data in same format as it came in as, unless `as_numpy` == True
        """
        if (np.array(self.stride) > np.array(self.kernel_size)).any():
            raise TesselatorException(
                "Stride is larger than kernel size, incoming data cannot be complete.\n"
                f"Set stride smaller, or use another tesselator "
            )

        if self._layout is None:
            raise TesselatorException(
                f"This tesselator has not be used to patch, therefore it has not recorded the layout, and cannot be used to stitch."
            )

        if not input_data.shape == self._post_shape and not self.ignore_difference:
            warnings.warn(
                f"Incoming shape is different to expected. {input_data.shape} != {self._post_shape}. This may fail to stitch the data back together.",
                TesselatorWarning,
            )

        all_patches = []
        for input_patch in input_data:
            all_patches.append(_patching.reorder.reorder(input_patch, data_format, "TCHW"))

        all_patches = np.array(all_patches)

        full_prediction = _patching.patches.rejoin_patches(
            _patching.patches.organise_patches(all_patches, factor_choice=self._layout),
            size=self.stride or self.kernel_size,
        )

        try:
            full_prediction = _patching.subset.center(full_prediction, self._initial_shape[-2:])
        except TesselatorException:
            warnings.warn(
                f"Could not trim to initial_shape, if 'padding' is None, an incomplete patch set is made. Padding with nans",
                TesselatorWarning,
            )
            pad_width = [
                *[(0, 0)] * (len(full_prediction.shape) - 2),
                *[
                    (0, self._initial_shape[-2 + i] - full_prediction.shape[-2 + i])
                    for i in range(len(full_prediction.shape[-2:]))
                ],
            ]
            full_prediction = np.pad(full_prediction, pad_width, constant_values=np.nan)

        if as_numpy or self._return_type == "numpy":
            return full_prediction

        dims, coords, attrs = self._get_coords()
        coords = dict(coords)
        attrs["description"] = "Reconstructed Data"

        if "Variables" in coords and var_select:
            coords["Variables"] = [coords["Variables"][var_select]]
        offset = [
            self.kernel_size[0] // 2 - self.stride[0] // 2,
            self.kernel_size[1] // 2 - self.stride[1] // 2,
        ]

        if offset[0] < 0 or offset[1] < 0:
            raise NotImplementedError("Calculated Offset is negative, which is currently not supported")

        if override:
            for override_key, override_value in override.items():
                if override_key in coords:
                    coords[override_key] = override_value

        if self.padding is None:
            coords[dims[-1]] = coords[dims[-1]][offset[0] : offset[0] + full_prediction.shape[-1]]
            coords[dims[-2]] = coords[dims[-2]][offset[1] : offset[1] + full_prediction.shape[-2]]

        if "Variables" in coords:
            variables = coords.pop("Variables")
            data_vars = {}

            if "time" in coords:
                coords["time"] = np.atleast_1d(coords["time"])
            for i in range(full_prediction.shape[dims.index("Variables")]):
                data = np.take(full_prediction, i, axis=dims.index("Variables"))
                data_vars[variables[i]] = (coords, data)

            ds = xr.Dataset(
                data_vars=data_vars,
                coords=coords,
                attrs=attrs.pop("global", {}),
            )

            for var in ds.data_vars:
                if var in self._attrs:
                    ds[var].attrs = self._attrs[var]

            return ds

        else:
            da = xr.DataArray(
                data=full_prediction,
                dims=dims,
                coords=coords,
                name=self.out_name,
                attrs=attrs,
            )
            return da.to_dataset()
