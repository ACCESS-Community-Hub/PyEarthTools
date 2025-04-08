import numpy as np
import pytest

from pyearthtools.utils.data.tesselator._patching.subset import center, cut_center
from pyearthtools.utils.exceptions import TesselatorException


def test_cut_center():
    x = np.zeros((10, 10))
    assert cut_center(x, 5).shape == (5, 5)
    assert cut_center(x, (6, 4)).shape == (6, 4)


def test_cut_center_square():
    x = np.zeros((10, 10))
    result = cut_center(x, 5)
    assert result.shape == (5, 5)


def test_cut_center_rectangle():
    x = np.zeros((10, 10))
    result = cut_center(x, (6, 4))
    assert result.shape == (6, 4)


def test_cut_center_large_size():
    x = np.zeros((10, 10))
    with pytest.raises(TesselatorException):
        cut_center(x, (11, 5))


def test_cut_center_invalid_size():
    x = np.zeros((10, 10))
    with pytest.raises(TesselatorException):
        cut_center(x, (6, 4, 2))


def test_cut_center_odd_grid_even_cut():
    """
    This test verifies that the cut_center function extracts a 4x4 subset
    from the bottom right quadrant of the 5x5 centroid of the grid.

    Input grid:
    [[  1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11],
     [ 12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22],
     [ 23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33],
     [ 34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44],
     [ 45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55],
     [ 56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66],
     [ 67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77],
     [ 78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88],
     [ 89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99],
     [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
     [111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121]]

     Center of the grid, as defined by the cut_center function:
     [[ 49,  50,  51,  52],
      [ 60,  61,  62,  63],
      [ 71,  72,  73,  74],
      [ 82,  83,  84,  85]]

    """

    x = np.arange(1, 122).reshape((11, 11))
    result = cut_center(x, 4)
    assert result.shape == (4, 4)
    expected_result = np.array([[49, 50, 51, 52], [60, 61, 62, 63], [71, 72, 73, 74], [82, 83, 84, 85]])
    assert np.array_equal(result, expected_result)


def test_cut_center_odd_grid_odd_cut():
    """
    This test verifies that the cut_center function extracts a 3x3 subset
    exactly from the centroid of the grid.
    """

    x = np.arange(1, 122).reshape((11, 11))
    result = cut_center(x, 3)
    assert result.shape == (3, 3)
    expected_result = np.array([[49, 50, 51], [60, 61, 62], [71, 72, 73]])
    assert np.array_equal(result, expected_result)


def test_cut_center_even_grid_even_cut():
    """
    This test verifies that the cut_center function extracts a 4x4 subset
    exactly from the centroid of the grid.
    """

    x = np.arange(1, 101).reshape((10, 10))
    result = cut_center(x, 4)
    assert result.shape == (4, 4)
    expected_result = np.array([[34, 35, 36, 37], [44, 45, 46, 47], [54, 55, 56, 57], [64, 65, 66, 67]])
    assert np.array_equal(result, expected_result)


def test_cut_center_even_grid_odd_cut():
    """
    This test verifies that the cut_center function extracts a 3x3 subset
    from the top left quadrant of the 4x4 centroid of the grid.

    Input grid:
    [[  1,   2,   3,   4,   5,   6,   7,   8,   9,  10],
     [ 11,  12,  13,  14,  15,  16,  17,  18,  19,  20],
     [ 21,  22,  23,  24,  25,  26,  27,  28,  29,  30],
     [ 31,  32,  33,  34,  35,  36,  37,  38,  39,  40],
     [ 41,  42,  43,  44,  45,  46,  47,  48,  49,  50],
     [ 51,  52,  53,  54,  55,  56,  57,  58,  59,  60],
     [ 61,  62,  63,  64,  65,  66,  67,  68,  69,  70],
     [ 71,  72,  73,  74,  75,  76,  77,  78,  79,  80],
     [ 81,  82,  83,  84,  85,  86,  87,  88,  89,  90],
     [ 91,  92,  93,  94,  95,  96,  97,  98,  99, 100]]

    Center of the grid, as defined by the cut_center function:
    [[45, 46, 47],
     [55, 56, 57],
     [65, 66, 67]]
    """
    x = np.arange(1, 101).reshape((10, 10))
    result = cut_center(x, 3)
    assert result.shape == (3, 3)
    expected_result = np.array([[34, 35, 36], [44, 45, 46], [54, 55, 56]])
    assert np.array_equal(result, expected_result)


def test_center_square():
    x = np.zeros((10, 1, 10, 10))

    result = center(x, 5)
    assert result.shape == (10, 1, 5, 5)


def test_center_rectangle():
    x = np.zeros((10, 1, 10, 10))
    result = center(x, (6, 4))
    assert result.shape == (10, 1, 6, 4)


def test_center():
    """
    Test the center function with input format "THWC".
    The center function first moves H and W to the end of the format.
    Then it takes the center of the last two dimensions.
    Then it reorders the center to the original format "THWC".
    """
    x = np.zeros((10, 10, 10, 1))  # Original format "THWC"
    result = center(x, (6, 4), "THWC")
    assert result.shape == (10, 6, 4, 1)
