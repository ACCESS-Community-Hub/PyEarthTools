import numpy as np
from persistence.methods._median import _median_of_three_numpy


def test_median_of_three_numpy_basic():
    """
    Tests that the dimensions are preserved except the time dimension which is
    reduced (but not squeezed) to one
    """

    # --- case 1 ---
    # create a simple array and throw in an outlier for sense check
    input_arr = np.array([[1, 2, 3], [5, 2, 6], [0, 191, 4]])
    expect_arr = np.array([[2], [5], [4]])
    idx_time = 1  # second dimension (idx=1) is time
    result_arr = _median_of_three_numpy(input_arr, idx_time)
    assert np.allclose(result_arr, expect_arr)

    # --- case 2 ---
    # check dimensionality is preserved for >2 dimensions
    # the values actually don't matter here.
    input_arr = np.full((5, 4, 3, 4, 5), 1, dtype=np.float64)
    idx_time = 3  # arbitrarily make fourth dimension time (idx_time = 3)
    expect_shape = (5, 4, 3, 1, 5)
    result_arr = _median_of_three_numpy(input_arr, idx_time)
    result_shape = result_arr.shape
    assert expect_shape == result_shape


def test_median_of_three_numpy_all_nans():
    """
    Test that all nans doesn't spit out a warning and that the associated
    dimension is filled with a `nan`
    """
    input_arr = np.array([[1, 2, 3], [5, 2, 6], [np.nan, np.nan, np.nan]])
    expect_arr = np.array([[2], [5], [np.nan]])
    idx_time = 1  # second dimension (idx=1) is time
    result_arr = _median_of_three_numpy(input_arr, idx_time)
    assert np.allclose(result_arr, expect_arr, equal_nan=True)


def test_median_of_three_numpy_partial_nan():
    """
    Test that partial nans are still handled. i.e. median of two numbers will
    just be their mean and median of one number will just be itself.
    """
    input_arr = np.array([[1, 2, 3], [5, 2, np.nan], [5, np.nan, np.nan]])
    expect_arr = np.array([[2], [3.5], [5]])
    idx_time = 1  # second dimension (idx=1) is time
    result_arr = _median_of_three_numpy(input_arr, idx_time)
    assert np.allclose(result_arr, expect_arr)
