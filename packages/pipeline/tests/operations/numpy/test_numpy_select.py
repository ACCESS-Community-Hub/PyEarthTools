# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
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

import pytest
import numpy as np


from pyearthtools.pipeline.operations.numpy import select


@pytest.fixture(scope="module")
def sample():
    """Test numpy array."""
    return np.array(range(24)).reshape((2, 3, 4))


def test_Select(sample):
    """Tests the Select numpy operation."""

    s = select.Select([0])

    output = s.apply_func(sample)

    assert output.shape == (3, 4)
    assert np.array_equal(output, sample[0, :, :])

    # multi-dimensional indexing
    s = select.Select([0, None, 3])

    output = s.apply_func(sample)

    assert output.shape == (3,)
    assert np.array_equal(output, sample[0, :, 3])

    # pass tuple of arrays
    output = s.apply_func((sample, sample))
    for arr in output:
        assert arr.shape == (3,)
        assert np.array_equal(arr, sample[0, :, 3])

    # pass tuple of arrays with tuple index
    s = select.Select(array_index=(0,), tuple_index=1)
    output = s.apply_func((sample, sample))
    assert output[0].shape == sample.shape
    assert np.array_equal(output[0], sample)
    assert output[1].shape == (3, 4)
    assert np.array_equal(output[1], sample[0])


def test_Slice(sample):
    """Tests the Slice numpy operations."""

    s = select.Slice((1,), (2,), (1, 4))
    output = s.apply_func(sample)
    assert output.shape == (1, 2, 3)
    assert np.array_equal(output, sample[:1, :2, 1:4])

    # test reverse_slice
    s = select.Slice((1,), (2,), reverse_slice=True)
    output = s.apply_func(sample)
    assert output.shape == (2, 1, 2)
    assert np.array_equal(output, sample[:, :1, :2])
