# Copyright Commonwealth of Australia, Bureau of Meteorology 2026.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pyearthtools.pipeline.operations.dask import reshape
from unittest.mock import MagicMock

import numpy as np
import dask.array as da
import pytest


def test_Rearrange():
    r = reshape.Rearrange("h l w -> h w l")
    h_dim = 2
    l_dim = 10
    w_dim = 20
    random_array = da.from_array(np.random.randn(h_dim, l_dim, w_dim))
    output = r.apply_func(random_array)
    undo_output = r.undo_func(output.compute())

    assert output.compute().shape == (h_dim, w_dim, l_dim), "Check dimensions rearranged correctly."
    assert undo_output.shape == random_array.shape, "Check undo successfully reverses."

    # undo_func also accepts a da.Array directly, computing it internally
    dask_output = da.from_array(output.compute())
    undo_from_dask = r.undo_func(dask_output)
    assert undo_from_dask.shape == random_array.shape, "Check undo works with dask input."


def test_Rearrange_explicit_reverse():
    """The undo can be detected automatically or given explicitly. This version tests what happens when it is
    given explicitly."""
    r = reshape.Rearrange("h l w -> l w h", reverse_rearrange="l w h -> h l w")
    h_dim = 1
    l_dim = 12
    w_dim = 6
    random_array = da.from_array(np.random.randn(h_dim, l_dim, w_dim))
    output = r.apply_func(random_array)
    undo_output = r.undo_func(output.compute())

    assert np.array_equal(undo_output, random_array.compute()), "Check explicit undo successfully reverses."


def test_Rearrange_skip():
    """Check that the operation can be skipped, if the skip flag is True."""
    r = reshape.Rearrange("h l w -> l w h", skip=True)
    h_dim = 1
    l_dim = 12
    wrong_shape_array = da.from_array(np.random.randn(h_dim, l_dim))
    output = r.apply_func(wrong_shape_array)

    assert np.array_equal(output.compute(), wrong_shape_array.compute()), "Check skip can leave array unchanged."


def test_Rearrange_not_skip():
    """Check that the operation can raise an error, if the skip flag is not set to True."""
    r = reshape.Rearrange("h l w -> l w h")
    h_dim = 1
    l_dim = 12
    wrong_shape_array = da.from_array(np.random.randn(h_dim, l_dim))
    with pytest.raises(Exception):
        r.apply_func(wrong_shape_array).compute()


def test_Squeeze():
    s = reshape.Squeeze(axis=(2, 3))
    random_array = da.from_array(np.random.randn(8, 8, 1, 1, 2, 1))
    output = s.apply_func(random_array)
    undo_output = s.undo_func(output)
    assert output.shape == (8, 8, 2, 1), "Squeeze only the correct axes."
    assert random_array.shape == undo_output.shape, "Check Squeeze can correctly undo itself."
    with pytest.raises(Exception):
        s.apply_func(output)  # Output doesn't have the correct axes of length 1, so we get an error.


def test_Expand():
    e = reshape.Expand(axis=(0, 2))
    random_array = da.from_array(np.random.randn(4, 3, 5))
    output = e.apply_func(random_array)
    undo_output = e.undo_func(output)
    assert output.shape == (1, 4, 1, 3, 5), "Expand the correct axes."
    assert undo_output.shape == random_array.shape, "Expand can undo itself."
    with pytest.raises(Exception):
        e.undo_func(random_array)


def test_Squeeze_reverses_Expand():
    e = reshape.Expand(axis=(0, 2))
    s = reshape.Squeeze(axis=(0, 2))
    random_array = da.from_array(np.random.randn(4, 3, 5))
    expand_output = e.apply_func(random_array)
    squeeze_output = s.apply_func(expand_output)
    assert squeeze_output.shape == random_array.shape, "Squeeze reverses Expand."


def test_Flattener():
    f = reshape.Flattener()
    random_array = da.from_array(np.random.randn(4, 3, 5))
    output = f.apply(random_array)
    undo_output = f.undo(output)
    assert len(output.shape) == 1, "Flattener produces a 1D array."
    assert np.array_equal(undo_output.compute(), random_array.compute()), "Flattener can undo itself."


def test_Flattener_1_dim():
    f2 = reshape.Flattener(flatten_dims=1)
    random_array = da.from_array(np.random.randn(4, 3, 5))
    output = f2.apply(random_array)
    undo_output = f2.undo(output)
    assert np.array_equal(output.compute(), random_array.compute()), "Flatten 1 dimension does nothing."
    assert np.array_equal(undo_output.compute(), random_array.compute()), "Undo Flatten 1 dimension."


def success_then_fail(self, *args, **kwargs):
    yield self
    raise ValueError()


def test_Flattener_exceptions():
    """Tests all the exceptions that can be raised in the Flattener class."""
    # try instantiating flattener with invalid dim
    with pytest.raises(ValueError):
        reshape.Flattener(flatten_dims=0)

    # test undo without apply
    f = reshape.Flattener(shape_attempt=(2, 1, 1))
    random_array = da.from_array(np.random.randn(4, 3, 5))
    with pytest.raises(RuntimeError):
        f.undo(random_array)

    # _configure_shape_attempt error when apply not run
    with pytest.raises(RuntimeError):
        f._configure_shape_attempt()

    # test undo when flatten_dims unset
    output = f.apply(random_array)
    f.flatten_dims = None  # "accidentally" overwrite the dims
    with pytest.raises(RuntimeError):
        f.undo(output)

    # setup flattener
    mock_array = MagicMock()
    mock_array.__len__.return_value = 1
    mock_array.shape = tuple([1])
    mock_array.reshape.return_value = mock_array
    f = reshape.Flattener()
    output = f.apply(mock_array)

    # trigger ValueError in undo when reshape fails
    mock_array.reshape.side_effect = ValueError
    with pytest.raises(ValueError):
        f.undo(mock_array)

    # error when input array shape not same rank as shape_attempt
    f = reshape.Flattener(shape_attempt=("...", 2))
    output = f.apply(random_array)
    with pytest.raises(IndexError):
        f.undo(output)


def test_Flatten():
    f1 = reshape.Flatten(flatten_dims=2)
    random_array = da.from_array(np.random.randn(4, 3, 5))
    output = f1.apply_func(random_array)
    undo_output = f1.undo_func(output)
    assert output.shape == (4, 3 * 5), "Flatten acts on the last few dimensions."
    assert np.array_equal(undo_output.compute(), random_array.compute()), "Flatten can undo itself."


def test_Flatten_1_dim():
    f2 = reshape.Flatten(flatten_dims=1)
    random_array = da.from_array(np.random.randn(4, 3, 5))
    output = f2.apply_func(random_array)
    undo_output = f2.undo_func(output)
    assert np.array_equal(output.compute(), random_array.compute()), "Flatten 1 dimension does nothing."
    assert np.array_equal(undo_output.compute(), random_array.compute()), "Undo Flatten 1 dimension."


def test_Flatten_all_dims():
    f3 = reshape.Flatten()
    random_array3 = da.from_array(np.random.randn(6, 7, 5, 2))
    output = f3.apply_func(random_array3)
    assert output.shape == (6 * 7 * 5 * 2,)
    assert f3.undo_func(output).shape == (6, 7, 5, 2), "Undo Flatten all dimensions."


def test_Flatten_with_shape_attempt():
    incoming_data = da.zeros((8, 1, 3, 3))
    f = reshape.Flatten(shape_attempt=(2, 1, 1, 1))
    f.apply_func(incoming_data)
    undo_data = da.zeros(2)
    assert f.undo_func(undo_data).shape == (2, 1, 1, 1)


def test_Flatten_with_shape_attempt_with_ellipses():
    incoming_data = da.zeros((8, 1, 3, 3))
    f = reshape.Flatten(shape_attempt=(2, "...", 1, 1))
    f.apply_func(incoming_data)
    undo_data = da.zeros(2)
    assert f.undo_func(undo_data).shape == (2, 1, 1, 1)


def test_Flatten_with_many_arrays():
    incoming_data = (da.zeros((8, 1, 3, 3)), da.zeros((8, 1, 3, 6)))
    f = reshape.Flatten()
    output = f.apply_func(incoming_data)
    assert isinstance(output, tuple)
    assert output[0].shape == (8 * 1 * 3 * 3,)
    assert output[1].shape == (8 * 1 * 3 * 6,)
    # undo
    output = f.undo(output)
    assert isinstance(output, tuple)
    assert output[0].shape == incoming_data[0].shape
    assert output[1].shape == incoming_data[1].shape


def test_SwapAxis():
    s = reshape.SwapAxis(1, 3)
    random_array = da.from_array(np.random.randn(5, 7, 8, 2))
    output = s.apply_func(random_array)
    assert output.shape == (5, 2, 8, 7), "Swap axes 1 and 3"
    undo_output = s.undo_func(output)
    assert np.array_equal(undo_output.compute(), random_array.compute()), "Undo axis swap."

    # undo_func also accepts numpy input directly
    numpy_output = output.compute()
    undo_numpy = s.undo_func(numpy_output)
    assert isinstance(undo_numpy, np.ndarray)
    assert np.array_equal(undo_numpy, random_array.compute()), "Undo axis swap with numpy input."


def test_Flattener_prod_shape_helper():
    """Tests the Flattener._prod_shape method with dask input."""
    f = reshape.Flattener()
    data = da.from_array(
        np.array(
            (
                (1, 2, 3),
                (4, 5, 6),
            )
        )
    )
    assert f._prod_shape(data) == 6  # product of data shape
