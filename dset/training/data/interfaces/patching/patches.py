"""
Patch related functions
"""

import math
from typing import Optional, Union

import numpy as np
import xarray as xr
from sklearn.feature_extraction import image

from dset.training.data.interfaces.patching import (
    DEFAULT_FORMAT_PATCH,
    DEFAULT_FORMAT_PATCH_AFTER,
    DEFAULT_FORMAT_PATCH_ORGANISE,
)
from dset.training.data.interfaces.patching.reorder import reorder
from dset.training.data.interfaces.patching.subset import cut_center


def factors(value: int) -> list[list[int, int]]:
    """
    Find factor Pairs of number

    Args:
        value: Number to find factor pairs of

    Returns:
        List of list of factor pairs
    """
    discovered_factors = []
    for i in range(1, int(value**0.5) + 1):
        if value % i == 0:
            discovered_factors.append([i, value // i])
    return discovered_factors


def organise_patches(
    patches: np.ndarray,
    axis_format: str = DEFAULT_FORMAT_PATCH_ORGANISE,
    factor_choice: Union[int, tuple[int, int]] = -1,
    invert: bool = False,
) -> np.ndarray:
    """
    Reorganise 1D list of patches, into 2D, Row-Column

    Finds factor pairs and reshapes array into such a pair
    Allows choice of which factor pair to use or user defined one, and if to invert

    Args:
        patches: Array of patches to organise
            Assumed to be of shape (Patches, ..., Width, Height)
            ... used as wildcard
        axis_format: Format of array, must be fully defined.
            Defaults to DEFAULT_FORMAT_PATCH_ORGANISE.
        factor_choice: Either int to choice factor pair, or tuple of factor pair.
            Defaults to -1.
        invert: Whether to invert factor pair before reshaping.
            Defaults to False.

    Raises:
        ValueError: If factor_choice as int is out of bounds of found factor pairs
        ValueError: If factor_choice is invalid type

    Returns:
        ndarray with Patches dimension split into Row and Column dimensions
    """
    patches = reorder(patches, axis_format, DEFAULT_FORMAT_PATCH_ORGANISE)

    num_patches = patches.shape[DEFAULT_FORMAT_PATCH_ORGANISE.find("P")]
    patch_factors = factors(num_patches)

    if isinstance(factor_choice, tuple):
        chosen_factor = list(factor_choice)

    elif isinstance(factor_choice, int):
        try:
            chosen_factor = patch_factors[factor_choice]
        except IndexError as exc:
            raise ValueError(
                f"Factor Choice '{factor_choice}' out of bounds of {patch_factors}"
            )
    else:
        raise ValueError(f"{type(factor_choice)} invalid for factor_choice")

    if invert:
        chosen_factor.reverse()

    patches = np.reshape(patches, (*chosen_factor, *patches.shape[1:]))

    patch_index = axis_format.find("P")
    axis_format = axis_format[:patch_index] + "R" + axis_format[patch_index:]

    patches = reorder(patches, DEFAULT_FORMAT_PATCH, axis_format)
    return patches


def rejoin_patches(
    patches: np.ndarray,
    size: Optional[Union[tuple[int, int], int]] = None,
    axis_format: str = DEFAULT_FORMAT_PATCH,
) -> np.ndarray:
    """
    Join patches together to form one coherent grid

    Note: see dset.datatools.subset.cut_center for center retrieval

    Args:
    ----------
        patches: Array of patches to rejoin
            Assumed to be of shape (Row, Patch, ..., Width, Height)
        size: Pixels to take from center of each patch
                E.g. for map of 256 pixels only stepping 128 (taking center) size = 128 or (128,128).
                Defaults to None / Take all.
        axis_format: String notating data axis arrangment. Defaults to 'RP...HW'
                (Row, Patch, ..., Width, Height)

    Returns:
    ----------
        Data without row and patch axis, as patches have been rejoined
        If custom axis_format given, format maintained without Row and Patch axes

    Examples:
    ----------
        >>> x = np.zeros([3,3,10,1,5,5])

        >>> rejoin_patches(x).shape
        (10, 1, 15, 15)

        >>> x = np.zeros([3,3,10,5,5])
        >>> rejoin_patches(x, size = 3).shape
        (10, 1, 9, 9)

        #(Time, Row, Patch, Width, Height, Channel)
        >>> x = np.zeros([10,3,3,5,5,1])
        >>> rejoin_patches(x, axis_format = "TRPHWC").shape
        (10, 15, 15, 1)
    """
    patches = reorder(patches, axis_format, DEFAULT_FORMAT_PATCH)

    datasize = (patches.shape[-2], patches.shape[-1])
    size = datasize[0] if size is None else size

    full_data = None
    for row in patches:
        rowbuild = None
        for patch in row:
            patch = cut_center(patch, size)

            rowbuild = (
                patch
                if rowbuild is None
                else np.concatenate((rowbuild, patch), axis=-1)
            )

        full_data = (
            rowbuild
            if full_data is None
            else np.concatenate((full_data, rowbuild), axis=-2)
        )

    axis_format = axis_format.replace("P", "")
    axis_format = axis_format.replace("R", "")
    full_data = reorder(full_data, DEFAULT_FORMAT_PATCH_AFTER, axis_format)

    return full_data


def make_patches(
    data,
    kernel_size: int,
    stride: int = None,
    data_format: str = None,
    padding="empty",
    **kwargs,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Split given data into a list of patches, maintaining all other dimension order


    Parameters
    ----------
    data
        Data to split
            Either - np.array or xr.Data[set/Array]
    kernel_size
        Size of patches to retrieve
    stride, optional
        Seperation between patches, by default None
            If None, = patch_size
    data_format, optional
        Ordering of dimensions for np.array input, by default None
    padding, optional
        Can be None to apply no padding
        str or function, by default 'empty'
        Must be one of np.pad valid modes
        'constant','empty','edge','wrap','reflect', etc

    Returns
    -------
        np array with patches as first dimension and height and width as last 2
        and number of patches in height and width made

    Raises
    ------
    TypeError
        If type of data not supported
    """

    data_as_array = None

    if isinstance(data, xr.Dataset):
        for var in list(data.data_vars):
            if data_as_array is None:
                data_as_array = data[var].to_numpy()
                data_as_array = np.expand_dims(data_as_array, axis=0)
            else:
                to_add = data[var].to_numpy()
                data_as_array = np.append(
                    data_as_array, np.expand_dims(to_add, axis=0), axis=0
                )
    elif isinstance(data, xr.DataArray):
        data_as_array = data.to_numpy()
        # data_as_array = np.expand_dims(data_as_array, axis=0)

    elif isinstance(data, np.ndarray):
        data_as_array = data

        if data_format:
            data_as_array = reorder(data_as_array, data_format, "CT...HW")
    else:
        raise TypeError(f"Data type '{type(data)} not supported")

    kernel_size = (
        [kernel_size, kernel_size]
        if isinstance(kernel_size, int)
        else list(kernel_size)
    )

    if stride is None:
        stride = kernel_size
    else:
        stride = [stride, stride] if isinstance(stride, int) else list(stride)

    if data_as_array is None:
        raise ValueError(f"Unable to convert input data into array: {data}")

    if padding is not None:
        padd_width = [(0, 0)] * (len(data_as_array.shape) - 2)

        def find_dim_expand(length, kernel, stride):
            if length % kernel == 0:
                return 0, 0
            result = ((((length // stride) + 1) * stride) - length) + (kernel - stride)
            return math.ceil(result / 2), math.floor(result / 2)

        padd_width.append(
            [*find_dim_expand(data_as_array.shape[-2], kernel_size[0], stride[0])]
        )
        padd_width.append(
            [*find_dim_expand(data_as_array.shape[-1], kernel_size[1], stride[1])]
        )
        # padd_width.append(((kernel_size[0] - stride[0])//2,(kernel_size[1] - stride[1])//2))

        if padding == "constant":
            kwargs["constant_values"] = kwargs.get("constant_values", np.nan)
        data_as_array = np.pad(
            data_as_array, pad_width=padd_width, mode=padding, **kwargs
        )

    first_dimensions = data_as_array.shape[:-2]

    kernel_size = (*first_dimensions, *kernel_size)
    stride = (*first_dimensions, *stride)

    patched = image._extract_patches(data_as_array, kernel_size, stride)

    layout = patched.shape[len(data_as_array.shape) - 2 : len(data_as_array.shape)]

    patched = patched.reshape((-1, *kernel_size))
    if data_format:
        patched = reorder(patched, "P...", "P" + data_format)

    return patched, layout
