from dataclasses import dataclass
import copy
import numpy as np
import warnings

import persistence.include._persistence_zig
from persistence.include._persistence_zig import ffi, lib


@dataclass(frozen=True)
class _MedianCommon:
    """
    This is a private namespace containing utility functions
    """

    def check_shape(arr_shape, idx_time):
        arr_shape = list(arr_shape)
        if arr_shape[idx_time] != 3:
            raise ValueError(
                "_median_of_three_numpy: the time dimension MUST only have 3 entries"
            )

    def get_output_shape(arr_shape, idx_time):
        arr_shape = list(arr_shape)
        arr_shape_out = copy.deepcopy(arr_shape)
        arr_shape_out[idx_time] = 1
        return arr_shape_out

    def check_and_convert_contiguousf32(arr: np.ndarray) -> np.ndarray:
        if not arr.flags["C_CONTIGUOUS"]:
            warnings.warn(
                UserWarning(
                    "_median_of_three_zig: input numpy array is not C contiguous! "
                    "Make sure to load the array using C contiguous settings. Focing to contiguous array."
                )
            )
            return np.ascontiguousarray(arr, dtype=np.float32, order="C")
        return arr.astype(np.float32)


def _median_of_three_numpy(arr: np.ndarray, idx_time: int) -> np.ndarray:
    """
    Computes median of three along the time index, preserves `nan`. IF a particular coordinate is all
    `nan` along the time dimension, THEN the output is `nan` for that entry.

    Uses numpy backend

    Returns the median of three applied along time dimension.

    IMPORTANT:
        - time dimension cardinality must equal 3

    Raises:
        ValueError: if time dimension does not have 3 entries
    """
    _MedianCommon.check_shape(arr.shape, idx_time)
    shape_out = _MedianCommon.get_output_shape(arr.shape, idx_time)
    # NOTE:
    #   - ignore numpy warnings as allowing all `nan` is intentional
    #   - `keepdims=True` because we want to keep the dimensional structure of the variable being
    #     computed at a higher level.
    #
    # FUTUREWORK:
    #   This should be replaced by a fast median of three algorithm using if/else statements or a
    #   ternary operator equivilent.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arr_median = np.nanmedian(arr, axis=idx_time, keepdims=True)
        assert list(shape_out) == list(arr_median.shape)
        return arr_median


def _median_of_three_zig(arr: np.ndarray, idx_time: int) -> np.ndarray:
    """
    Computes median of three along the time index, preserves `nan`. IF a particular coordinate is all
    `nan` along the time dimension, THEN the output is `nan` for that entry.

    Uses zig backend

    Returns the median of three applied time dimension.

    IMPORTANT:
        - input array (assumed to be a chunk) should be reasonably sized so that it doesn't need to
          call the FFI multiple times per chunk.
        - the input array must be C-contiguous, otherwise it'll be forced to be C-contiguous,
          requiring an extra copy operation.
        - time dimension cardinality must equal 3

    PERFORMANCE:
        This function performs best if:
        - the input array/chunk it deals with is large
        - the array is already in float32, most of the time xarray presents in float64
          (aside: float32 is more than enough for median calculations given that it's mainly a
          sorting algorithm, and doesn't introduce additional error.)
        - the array is already C-contiguous
        - the above would mean that most of the work is done in zig, and not in converting the array
          to conform.

    Raises:
        ValueError: if time dimension has more than 3 entries
        UserWarning: if array is not C contiguous
    """
    # check/transform input to conform to c-types
    _MedianCommon.check_shape(arr.shape, idx_time)
    arr32_in = _MedianCommon.check_and_convert_contiguousf32(arr)
    shape_out = _MedianCommon.get_output_shape(arr.shape, idx_time)
    shape_in = np.array(arr.shape, dtype=np.int32, order="C")
    arr32_out = np.empty(shape_out, dtype=np.float32, order="C")

    # gather inputs to pass to cffi
    # --- not sure if this is optimal ---
    ptr_arr32_in = ffi.from_buffer("float[]", arr32_in)
    ptr_arr32_out = ffi.from_buffer("float[]", arr32_out)
    ptr_shape_in = ffi.from_buffer("int[]", shape_in)
    # ---
    len_shape_in = len(shape_in)
    len_in = arr32_in.size
    len_out = arr32_out.size

    # safety
    assert isinstance(len_in, int)
    assert isinstance(len_out, int)

    lib.median_of_three_nd(
        int(idx_time),
        ptr_shape_in,
        len_shape_in,
        ptr_arr32_in,
        int(len_in),
        ptr_arr32_out,
        int(len_out),
    )

    # revert to original array type
    return arr32_out.astype(arr.dtype)
