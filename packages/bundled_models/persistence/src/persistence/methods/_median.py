import numpy as np
import warnings

# TODO: get this from common definition - requires refactor
_LOOKBACK = 3


def _median_of_three_numpy(arr: np.ndarray, idx_time: int) -> np.ndarray:
    """
    Computes median of three along the time index, ignores nans; if a
    particular coordinate is all nan for the required time indices, the
    output is nan for that entry.
    """
    # safety: this should have been handled at the top level
    len_time = arr.shape[idx_time]
    assert len_time >= _LOOKBACK

    # ---
    # select relenvant array indices by time, based on lookback
    #
    # TODO: this should happen someplace higher up assumes latest obs is at the end, similar to
    #       _LOOKBACK.
    idx_end = len_time
    idx_start = idx_end - _LOOKBACK
    idx_slice = slice(idx_start, idx_end, 1)  # start, end, step
    # generator for nd-index slicing
    idx_all = slice(None, None, None)
    nd_slice = (idx_slice if i == idx_time else idx_all for i in range(len(arr.shape)))
    # sliced array that only has the latest 3 values
    arr_slice = arr[*tuple(nd_slice)]
    # ---

    # ---
    # calculate the median along the time axis
    #
    # NOTE: ignore numpy warnings as allowing all `nan` is intentional
    #
    # NOTE: `keepdims=True` because we want to keep the dimensional structure of the variable
    #       being computed at a higher level.
    #
    # TODO: this should be replaced by a fast median of three algorithm using if/else statements
    #       or a ternary operator equivilent.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arr_median = np.nanmedian(arr_slice, axis=idx_time, keepdims=True)
        return arr_median
    # ---
