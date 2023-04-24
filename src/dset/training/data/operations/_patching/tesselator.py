"""
Tesselation Tool
Provides ways to split data into patches, and reform patches back to full data
"""

import logging
from typing import Iterable, Union

import numpy as np
import xarray as xr

from . import _patching


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
        coord_template=None,
        out_name: str = "Reconstructed",
    ):
        """Create Tesselator

        Used to split data into patches and to reform patches

        Args:
            kernel_size (int): Size of each individual kernel
            stride (int, optional): Distance between kernels.If none, set to kernel_size. Defaults to None.
            padding (str, optional): Padding operation, either str or function. Must be one of np.pad modes. Defaults to "reflect".
            coord_template (_type_, optional): Set coordinate template for stitch output. Defaults to None.
            out_name (str, optional): Name of dataArray outputted from stitch. Defaults to "Reconstructed".
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
        self._dims = None
        self._attrs = {}

        self.out_name = out_name
        
        if coord_template:
            self._set_coords(coord_template)
        
        self._initial_shape = None
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
        elif isinstance(data,  xr.DataArray):
            shape = data.shape
        else:
            raise TypeError(f"Unable to find shape of {data!r}")
        return shape

    def _set_coords(self, data: Union[xr.DataArray, xr.Dataset,  np.ndarray]):
        """
        From an xr DataArray or Dataset, save coordinates and dims for stitching

        Parameters
        ----------
        data
            Template data to get coords and dims from
        """

        shape = self._find_shape(data)

        if self._initial_shape and not shape == self._initial_shape:
            raise RuntimeError(f"Initial shape was {self._initial_shape!r} which doesn't match incoming shape {shape!r}")
        else:
            self._initial_shape = shape

        if isinstance(data, np.ndarray) or data is None:
            self._return_type = 'numpy'
            return

        elif isinstance(data, (xr.Dataset, xr.DataArray)):
                
            self._attrs['global'] = data.attrs
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

    def patch(
        self,
        input_data: Union[xr.DataArray, xr.Dataset,  np.ndarray],
        data_format: str = None,
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

        patches, layout = _patching.patches.make_patches(
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
                _patching.reorder.reorder(input_patch, data_format, "TCHW")
            )

        all_patches = np.array(all_patches)

        full_prediction = _patching.patches.rejoin_patches(
            _patching.patches.organise_patches(all_patches, factor_choice=self._layout),
            size=self.stride or self.kernel_size,
        )

        if self.padding and self._initial_shape:
            full_prediction = _patching.subset.center(
                full_prediction, self._initial_shape[-2:]
            )

        if as_numpy or self._return_type == 'numpy':
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
        
        if self.padding is None:
            coords[dims[-1]] = coords[dims[-1]][
                offset[0] : offset[0] + full_prediction.shape[-1]
            ]
            coords[dims[-2]] = coords[dims[-2]][
                offset[1] : offset[1] + full_prediction.shape[-2]
            ]

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
                attrs=attrs.pop('global', {}),
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
