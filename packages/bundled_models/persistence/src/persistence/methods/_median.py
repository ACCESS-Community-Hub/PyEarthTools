import numpy as np
import warnings

import persistence.include._persistence_zig
from persistence.include._persistence_zig import ffi, lib

def _median_of_three_numpy(arr: np.ndarray, idx_time: int) -> np.ndarray:
    """
    Computes median of three along the time index, preserves `nan`. IF a particular coordinate is all
    `nan` along the time dimension, THEN the output is `nan` for that entry.
    """
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
        return arr_median


# --- TODO: for testing only
def _median_of_three_zig(x1: int, x2: int, x3: int) -> int:
    y = lib.median_of_three(x1, x2, x3)
    return y
# ---
