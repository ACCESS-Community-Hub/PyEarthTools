"""
Tesselation Layer for Pipeline
Provides ways to split data into patches, and reform patches back to full data
"""

import logging
from typing import Iterable, Union

import numpy as np
import xarray as xr

from dset.training.data.indexes import patching


def tuple_difference(tuple_1, tuple_2):
    """
    Find elements missing from the second tuple which are included in the first
    """
    list_1 = list(tuple_1)
    list_2 = list(tuple_2)

    for element in list_2:
        if element in list_1:
            list_1.remove(element)
    return tuple(list_1)


class Tesselator:
    def __init__(
        self,
        kernel_size: int,
        stride: int = None,
        padding: str = "reflect",
        coord_template=None,
        out_name: str = "Reconstructed",
    ):
        """
        Initialise Tesselator Layer of Pipeline

        Used to split data into patches and to reform patches

        Parameters
        ----------
        kernel_size
            Size of each individual kernel
        stride, optional
            Distance between kernels, by default None
                If none, set to kernel_size
        padding, optional
            Padding operation, either str or function, by default 'reflect'
                Must be one of np.pad modes
        coord_template, optional
            Set coordinate template for stitch output, by default None
        out_name, optional
            Name of dataArray outputted from stitch, by default "Reconstructed"
        """
        self.kernel_size = (
            [kernel_size] if isinstance(kernel_size, int) else list(kernel_size)
        )

        if len(self.kernel_size) == 1:
            self.kernel_size = self.kernel_size * 2

        self.stride = stride or kernel_size
        self.stride = (
            [self.stride] if isinstance(self.stride, int) else list(self.stride)
        )
        if len(self.stride) == 1:
            self.stride = self.stride * 2

        self.padding = padding

        self._coords = None
        self.out_name = out_name
        self._set_coords(coord_template)
        self._initial_shape = None
        self._layout = None

    def _set_coords(self, data: Union[xr.DataArray, xr.Dataset]):
        """
        From a xr DataArray or Dataset, save coordinates and dims for stitching

        Parameters
        ----------
        data
            Template data to get coords and dims from
        """
        if isinstance(data, (xr.DataArray, xr.Dataset)):
            self._dims = list(data.coords)
            if "time" in self._dims:
                self._dims.remove("time")
                self._dims.append("time")
            self._coords = {}
            self._attrs = data.attrs

            for dim in self._dims:
                self._coords[dim] = data[dim].values

            if isinstance(data, xr.Dataset):
                self._variables = list(data.data_vars)
                self._initial_shape = (
                    len(self._variables),
                    *data[self._variables[0]].shape,
                )

                self._dims = ["Variables"] + self._dims
                self._coords["Variables"] = self._variables

            else:
                self._initial_shape = data.shape

    def _get_coords(self) -> tuple[list, dict]:
        """
        Retrieve coords and dims from self

        Returns
        -------
            Dims, Coordinates

        Raises
        ------
        AttributeError
            If no template has been provided
        """
        if self._coords is None:
            raise AttributeError(
                "No template has been provided, unable to assign coordinates"
            )

        return list(self._dims), dict(self._coords), self._attrs

    def _organise_coords(
        self, dims: tuple, coords: dict, data: np.ndarray
    ) -> tuple[list, dict]:
        new_dims = []
        new_coords = {}

        if len(data.shape) + 1 == len(coords):
            coords_shape = tuple(
                len(value) if isinstance(value, Iterable) else 1
                for key, value in coords.items()
            )

            if 1 in tuple_difference(coords_shape, data.shape):
                data = np.expand_dims(data, 0)

        for i, axis_size in enumerate(data.shape):
            found_match = False
            for key, value in coords.items():
                if not isinstance(value, (Iterable, str)):
                    value = [value]

                if len(value) == axis_size:
                    if key in new_coords:
                        continue
                    found_match = True
                    new_dims.append(key)
                    new_coords[key] = value
                    break

            if not found_match:
                logging.warn(f"Adding Fake Coordinate at location {i}")
                new_coords[f"Coordinate {i}"] = list(range(0, axis_size))
                new_dims.append(f"Coordinate {i}")

        return new_dims, new_coords, data

    def patch(
        self,
        input_data: xr.Dataset,
        data_format: str = None,
        update: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        From a data source, get patches

        Parameters
        ----------
        input_data
            DataSource
        data_format
            Format of data, if incoming data is np.array, by default None

        Yields
        ------
            Each Patch as np.ndarray, size defined in __init__
        """
        self._set_coords(input_data)

        patches, layout = patching.patches.make_patches(
            input_data,
            self.kernel_size,
            self.stride,
            data_format=data_format,
            padding=self.padding,
            **kwargs,
        )

        self._layout = layout

        return patches

    def stitch(
        self,
        input_data: np.ndarray,
        data_format: str = "TCHW",
        override=None,
        var_select=None,
        as_numpy: bool = False,
    ):
        """
        Stitch patches back together and assign coordinates

        Requires that a template has been provided

        Parameters
        ----------
        input_data
            Input Patches
        data_format, optional
            Order of dimensions in input_data, by default "TCHW"

        Yields
        ------
            Reshaped and formatted data

        Raises
        ------
        NotImplementedError
            If offset required to align coordinates is negative
                Typically occurs if step_size > patch_size
        """

        all_patches = []
        for input_patch in input_data:
            all_patches.append(
                patching.reorder.reorder(input_patch, data_format, "TCHW")
            )

        all_patches = np.array(all_patches)

        full_prediction = patching.patches.rejoin_patches(
            patching.patches.organise_patches(all_patches, factor_choice=self._layout),
            size=self.stride or self.kernel_size,
        )

        if self._initial_shape is None or as_numpy:
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
            raise NotImplementedError(
                "Calculated Offset is negative, which is currently not supported"
            )

        if override:
            for override_key, override_value in override.items():
                if override_key in coords:
                    coords[override_key] = override_value

        if self.padding is not None:
            full_prediction = patching.subset.center(
                full_prediction, self._initial_shape[-2:]
            )
            dims, coords, full_prediction = self._organise_coords(
                dims, coords, full_prediction
            )
        else:
            coords[dims[-1]] = coords[dims[-1]][
                offset[0] : offset[0] + full_prediction.shape[-1]
            ]
            coords[dims[-2]] = coords[dims[-2]][
                offset[1] : offset[1] + full_prediction.shape[-2]
            ]

        if "Variables" in coords:
            variables = coords.pop("Variables")
            data_vars = {}
            # coords.pop('time')

            if "time" in coords:
                coords["time"] = np.atleast_1d(coords["time"])
            for i in range(full_prediction.shape[dims.index("Variables")]):
                data = np.take(full_prediction, i, axis=dims.index("Variables"))
                data_vars[variables[i]] = (coords, data)

            ds = xr.Dataset(
                data_vars=data_vars,
                coords=coords,
                attrs=attrs,
            )
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
