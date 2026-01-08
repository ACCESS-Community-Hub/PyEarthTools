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
import dask.array as da


from pyearthtools.pipeline.operations.dask import select


@pytest.fixture(scope="module")
def sample():
    """Test dask array."""
    return da.array(range(24)).reshape((2, 3, 4))


def test_Select(sample):
    """Tests the Select dask operation."""

    s = select.Select([0])

    output = s.apply_func(sample)

    assert output.shape == (3, 4)
    assert (output == sample[0, :, :]).all().compute()

    # multi-dimensional indexing
    s = select.Select([0, None, 3])

    output = s.apply_func(sample)

    assert output.shape == (3,)
    assert (output == sample[0, :, 3]).all().compute()

    # pass tuple of arrays
    output = s.apply_func((sample, sample))
    for arr in output:
        assert arr.shape == (3,)
        assert (arr == sample[0, :, 3]).all().compute()

    # pass tuple of arrays with tuple index
    s = select.Select(array_index=(0,), tuple_index=1)
    output = s.apply_func((sample, sample))
    assert output[0].shape == sample.shape
    assert (output[0] == sample).all().compute()
    assert output[1].shape == (3, 4)
    assert (output[1] == sample[0]).all().compute()


def test_Slice(sample):
    """Tests the Slice dask operation."""

    s = select.Slice((1,), (2,), (1, 4))
    output = s.apply_func(sample)
    assert output.shape == (1, 2, 3)
    assert (output == sample[:1, :2, 1:4]).all().compute()

    # test reverse_slice
    s = select.Slice((1,), (2,), reverse_slice=True)
    output = s.apply_func(sample)
    assert output.shape == (2, 1, 2)
    assert (output == sample[:, :1, :2]).all().compute()
