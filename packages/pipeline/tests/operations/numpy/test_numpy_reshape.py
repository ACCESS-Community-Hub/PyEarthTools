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
    r = reshape.Rearrange('h l w -> h w l')
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
    r = reshape.Rearrange('h l w -> l w h', reverse_rearrange='l w h -> h l w')
    h_dim = 1
    l_dim = 12
    w_dim = 6
    random_array = np.random.randn(h_dim, l_dim, w_dim)
    output = r.apply_func(random_array)
    undo_output = r.undo_func(output)

    assert np.all(undo_output == random_array), "Check explicit undo successfully reverses."

def test_Rearrange_skip():
    """Check that the operation can be skipped, if the skip flag is True."""
    r = reshape.Rearrange('h l w -> l w h', skip=True)
    h_dim = 1
    l_dim = 12
    wrong_shape_array = np.random.randn(h_dim, l_dim)
    output = r.apply_func(wrong_shape_array)

    assert np.all(output == wrong_shape_array), "Check skip can leave array unchanged."

def test_Rearrange_not_skip():
    """Check that the operation is not skipped, if the skip flag is not set to True."""
    r = reshape.Rearrange('h l w -> l w h')
    h_dim = 1
    l_dim = 12
    wrong_shape_array = np.random.randn(h_dim, l_dim)
    with pytest.raises(Exception):
        r.apply_func(wrong_shape_array)


def test_Squeeze():
    s = reshape.Squeeze(axis=(2, 3))
    random_array = np.random.randn(8, 8, 1, 1, 2, 1)
    assert s.apply_func(random_array).shape == (8, 8, 2, 1), "Squeeze only the correct axes."

def test_undo_Squeeze():
    s = reshape.Squeeze(axis=(2, 3))
    random_array = np.random.randn(8, 8, 1, 1, 2, 1)
    output = s.apply_func(random_array)
    undo_output = s.undo_func(output)
    assert random_array.shape == undo_output.shape, "Check Squeeze can correctly undo itself."

def test_Squeeze_error():
    """Check we get an error if we try to squeeze an axis not of length 1."""
    s = reshape.Squeeze(axis=(1, 3)) # Note axis 1, below, is not of length 1.
    random_array = np.random.randn(8, 8, 1, 1, 2, 1)
    with pytest.raises(Exception):
        s.apply_func(random_array)


