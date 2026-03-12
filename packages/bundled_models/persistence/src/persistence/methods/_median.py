import numpy as np
import warnings


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
