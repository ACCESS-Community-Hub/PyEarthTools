# Copyright Commonwealth of Australia, Bureau of Meteorology 2025.
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

from pyearthtools.pipeline.operations.numpy import reshape

import numpy as np
import pytest


def test_Rearrange():
    r = reshape.Rearrange("h l w -> h w l")
    h_dim = 2
    l_dim = 10
    w_dim = 20
    random_array = np.random.randn(h_dim, l_dim, w_dim)
    output = r.apply_func(random_array)
    undo_output = r.undo_func(output)

    assert output.shape == (h_dim, w_dim, l_dim), "Check dimensions rearranged correctly."
    assert np.all(undo_output.shape == random_array.shape), "Check undo successfully reverses."


def test_Rearrange_explicit_reverse():
    """The undo can be detected automatically or given explicitly. This version tests what happens when it is
    given explicitly."""
    r = reshape.Rearrange("h l w -> l w h", reverse_rearrange="l w h -> h l w")
    h_dim = 1
    l_dim = 12
    w_dim = 6
    random_array = np.random.randn(h_dim, l_dim, w_dim)
    output = r.apply_func(random_array)
    undo_output = r.undo_func(output)

    assert np.all(undo_output == random_array), "Check explicit undo successfully reverses."


def test_Rearrange_skip():
    """Check that the operation can be skipped, if the skip flag is True."""
    r = reshape.Rearrange("h l w -> l w h", skip=True)
    h_dim = 1
    l_dim = 12
    wrong_shape_array = np.random.randn(h_dim, l_dim)
    output = r.apply_func(wrong_shape_array)

    assert np.all(output == wrong_shape_array), "Check skip can leave array unchanged."


def test_Rearrange_not_skip():
    """Check that the operation can raise an error, if the skip flag is not set to True."""
    r = reshape.Rearrange("h l w -> l w h")
    h_dim = 1
    l_dim = 12
    wrong_shape_array = np.random.randn(h_dim, l_dim)
    with pytest.raises(Exception):
        r.apply_func(wrong_shape_array)


def test_Squeeze():
    s = reshape.Squeeze(axis=(2, 3))
    random_array = np.random.randn(8, 8, 1, 1, 2, 1)
    output = s.apply_func(random_array)
    undo_output = s.undo_func(output)
    assert output.shape == (8, 8, 2, 1), "Squeeze only the correct axes."
    assert random_array.shape == undo_output.shape, "Check Squeeze can correctly undo itself."
    with pytest.raises(Exception):
        s.apply_func(output)  # Output doesn't have the correct axes of length 1, so we get an error.


def test_Expand():
    e = reshape.Expand(axis=(0, 2))
    random_array = np.random.randn(4, 3, 5)
    output = e.apply_func(random_array)
    undo_output = e.undo_func(output)
    assert output.shape == (1, 4, 1, 3, 5), "Expand the correct axes."
    assert undo_output.shape == random_array.shape, "Expand can undo itself."
    with pytest.raises(Exception):
        e.undo_func(random_array)


def test_Squeeze_reverses_Expand():
    e = reshape.Expand(axis=(0, 2))
    s = reshape.Squeeze(axis=(0, 2))
    random_array = np.random.randn(4, 3, 5)
    expand_output = e.apply_func(random_array)
    squeeze_output = s.apply_func(expand_output)
    assert squeeze_output.shape == random_array.shape, "Squeeze reverses Expand."


def test_Flattener():
    f = reshape.Flattener()
    random_array = np.random.randn(4, 3, 5)
    output = f.apply(random_array)
    undo_output = f.undo(output)
    assert len(output.shape) == 1, "Flattener produces a 1D array."
    assert np.all(undo_output == random_array), "Flattener can undo itself."


def test_Flattener_1_dim():
    f2 = reshape.Flattener(flatten_dims=1)
    random_array = np.random.randn(4, 3, 5)
    output = f2.apply(random_array)
    undo_output = f2.undo(output)  # Check that the undo still works.
    assert np.all(output == random_array), "Flatten 1 dimension does nothing."
    assert np.all(undo_output == random_array), "Undo Flatten 1 dimension."


def test_Flatten():
    f1 = reshape.Flatten(flatten_dims=2)
    random_array = np.random.randn(4, 3, 5)
    output = f1.apply_func(random_array)
    undo_output = f1.undo_func(output)
    assert output.shape == (4, 3 * 5), "Flatten acts on the last few dimensions."
    assert np.all(undo_output == random_array), "Flatten can undo itself."


def test_Flatten_1_dim():
    f2 = reshape.Flatten(flatten_dims=1)
    random_array = np.random.randn(4, 3, 5)
    output = f2.apply_func(random_array)
    undo_output = f2.undo_func(output)  # Check that the undo still works.
    assert np.all(output == random_array), "Flatten 1 dimension does nothing."
    assert np.all(undo_output == random_array), "Undo Flatten 1 dimension."


def test_Flatten_all_dims():
    f3 = reshape.Flatten()
    random_array3 = np.random.randn(6, 7, 5, 2)
    output = f3.apply_func(random_array3)
    assert output.shape == (6 * 7 * 5 * 2,)
    assert f3.undo_func(output).shape == (6, 7, 5, 2), "Undo Flatten all dimensions."


def test_Flatten_with_shape_attempt():
    incoming_data = np.zeros((8, 1, 3, 3))
    f = reshape.Flatten(shape_attempt=(2, 1, 1, 1))
    f.apply_func(incoming_data)
    undo_data = np.zeros(2)
    assert f.undo_func(undo_data).shape == (2, 1, 1, 1)


def test_Flatten_with_shape_attempt_with_ellipses():
    incoming_data = np.zeros((8, 1, 3, 3))
    f = reshape.Flatten(shape_attempt=(2, "...", 1, 1))
    f.apply_func(incoming_data)
    undo_data = np.zeros(2)
    assert f.undo_func(undo_data).shape == (2, 1, 1, 1)


def test_SwapAxis():
    s = reshape.SwapAxis(1, 3)
    random_array = np.random.randn(5, 7, 8, 2)
    output = s.apply_func(random_array)
    assert output.shape == (5, 2, 8, 7), "Swap axes 1 and 3"
    undo_output = s.undo_func(output)
    assert np.all(undo_output == random_array), "Undo axis swap."
