import numpy as np
import pytest

from pyearthtools.utils.data.tesselator._patching._reorder import (
    move_to_end,
    reorder,
    setup_formats,
)


# Test setup_formats function.
def test_setup_formats():
    # Test with one fully defined format and one partially defined format.
    assert setup_formats("RPTCHW", "RP...HW") == ("RPTCHW", "RPTCHW")
    assert setup_formats("TCRPHW", "RP...HW") == ("TCRPHW", "RPTCHW")
    assert setup_formats("RP...HW", "TCRPHW") == ("RPTCHW", "TCRPHW")

    # Test with both formats fully defined.
    assert setup_formats("RPTCHW", "RPTCHW") == ("RPTCHW", "RPTCHW")
    assert setup_formats("TCRPHW", "RPTCHW") == ("TCRPHW", "RPTCHW")

    # Test with different combinations of characters in the formats.
    assert setup_formats("RP...HW", "RPXYHW") == ("RPXYHW", "RPXYHW")


def test_setup_formats_value_error():
    with pytest.raises(ValueError):
        # Test with both formats partially defined (should raise ValueError).
        setup_formats("RP...HW", "RP...HW")


# Test reorder function.
def test_reorder_same_format():
    data = np.zeros((3, 4, 5))
    result = reorder(data, "TAC", "TAC")
    np.testing.assert_array_equal(result, data)

    data = np.zeros((3, 4, 5, 6))  # Test with same longer formats.
    result = reorder(data, "CTHW", "CTHW")
    np.testing.assert_array_equal(result, data)


def test_reorder_different_formats():
    data = np.zeros((3, 4, 5))
    result = reorder(data, "TAC", "CAT")
    assert result.shape == (5, 4, 3)

    result = reorder(data, "TAC", "CTA")
    assert result.shape == (5, 3, 4)

    # Test with different longer formats.
    data = np.zeros((3, 4, 5, 6))
    result = reorder(data, "CTHW", "WHTC")
    assert result.shape == (6, 5, 4, 3)

    result = reorder(data, "CTHW", "WCTH")
    assert result.shape == (6, 3, 4, 5)


def test_reorder_invalid_format():
    data = np.zeros((3, 4, 5))
    with pytest.raises(ValueError):
        reorder(data, "TAC", "CT")


# Test move_to_end function.
def test_move_to_end_single_axis():
    data = np.zeros((3, 4, 5))
    new_format, reordered_data = move_to_end(data, "TAC", "T")
    assert new_format == "ACT"
    assert reordered_data.shape == (4, 5, 3)


def test_move_to_end_multiple_axes():
    data = np.zeros((3, 4, 5))
    new_format, reordered_data = move_to_end(data, "TAC", "TA")
    assert new_format == "CTA"
    assert reordered_data.shape == (5, 3, 4)


def test_move_to_end_all_axes():
    # Test with all axes, i.e. moving all the data.
    data = np.zeros((3, 4, 5))
    new_format, reordered_data = move_to_end(data, "TAC", "AT")
    assert new_format == "CAT"
    assert reordered_data.shape == (5, 4, 3)


def test_move_to_end_invalid_axis():
    data = np.zeros((3, 4, 5))
    with pytest.raises(KeyError):
        move_to_end(data, "TAC", "X")
