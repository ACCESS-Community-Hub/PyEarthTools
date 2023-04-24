"""
Apply subset filters to data
"""

import math
from typing import Iterable, Union

import numpy as np

from . import DEFAULT_FORMAT_SUBSET
from .reorder import move_to_end, reorder


def cut_center(data: np.ndarray, size: Union[int, tuple[int, int]]) -> np.ndarray:
    """
    Retrieve region from last 2 dimensions of array

    Note: if odd size, lower index round down, upper index round up
    Lower and Lower

    Args:
    ----------
        data: The data to retrieve center of
        size: Size of retrieval zone, either int for height & width
            or tuple for Height & Width each
    Raises:
    ----------
        ValueError: If desired size is larger than data size
        ValueError: If type of size not recognised

    Returns:
    ----------
        Data with subset of Height and Width

    Examples:
    ----------
        >>> x = np.zeros((10, 10))
        >>> cut_center(x, 5).shape
        (5, 5)

        >>> cut_center(x, (6, 4)).shape
        (6, 4)
    """
    center_values = (math.ceil(data.shape[-2] / 2), math.ceil(data.shape[-1] / 2))

    if isinstance(size, int):
        size = (size, size)

    if size[0] > data.shape[-2] or size[1] > data.shape[-1]:
        raise ValueError(
            f"Unable to trim data to large then its size." f"{size} > {data.shape[-2:]}"
        )

    if isinstance(size, Iterable) and len(size) == 2:
        trim_offset = (int(size[0]) / 2, int(size[1]) / 2)
    else:
        raise ValueError(f"{type(size)} not supported, must be int, or tuple(int,int)")
    
    data = data[
        ...,
        center_values[0]
        - math.ceil(trim_offset[0])  : center_values[0]
        + math.floor(trim_offset[0]),
        center_values[1]
        - math.ceil(trim_offset[1])  : center_values[1]
        + math.floor(trim_offset[1]),
    ]

    return data


def center(
    data: np.ndarray,
    size: Union[int, tuple[int, int]],
    axis_format: str = DEFAULT_FORMAT_SUBSET,
) -> np.ndarray:
    """
    Retrieve region from center of data array.
    Assumes data to have Height & Width as last two dims

    Use format if Height, Width of data is not the last two axis.


    Args:
    ----------
        data: The data to retrieve center of
        size: Size of retrieval zone, either int for height & width
            or tuple for Height & Width each
        axis_format: Arrangement of axis. Defaults to DEFAULT_FORMAT_SUBSET.
            If other axis_format is used, input axis arrangment will be maintained

    Raises:
    ----------
        ValueError: If type of size not recognised

    Returns:
    ----------
        Data with subset of Height and Width axis

    Examples:
    ----------
        >>> x = np.zeros((10, 1, 10, 10))

        >>> center(x, 5).shape
        (10, 1, 5, 5)

        >>> x = np.zeros((10, 10, 10))
        >>> center(x, (6, 4)).shape
        (10, 6, 4)

        >>> x = np.zeros((10, 10, 10, 1)) #THWC
        >>> center(x, (6, 4), "THWC").shape
        (10, 6, 4, 1)

    """

    altered_format, data = move_to_end(
        data, axis_format, "HW"
    )  # Change to known format

    data = cut_center(data, size)

    data = reorder(data, altered_format, axis_format)  # Set back to input format

    return data
