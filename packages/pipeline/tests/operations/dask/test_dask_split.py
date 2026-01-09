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

from pyearthtools.pipeline.operations.dask import split

import dask.array as da
import pytest


@pytest.fixture(scope="module")
def example_data():
    return da.array(range(2 * 3 * 4)).reshape((2, 3, 4))


def test_onaxis(example_data):
    """Tests dask OnAxis split operation class."""
    op = split.OnAxis(axis=1)

    # try join before splitting
    with pytest.raises(RuntimeError):
        op.join((example_data, example_data))
    result = op.split(example_data)
    assert all((arr == example_data[:, d, :]).all().compute() for d, arr in enumerate(result))

    orig = op.join(result)
    assert (orig == example_data).all().compute()


def test_onslice(example_data):
    """Tests dask OnSlice split operation class."""
    slices = ((0, 1), (1, 2), (2, 4))
    op = split.OnSlice(*slices, axis=2)
    result = op.split(example_data)
    for sl, arr in zip(slices, result, strict=True):
        assert (arr == example_data[:, :, sl[0] : sl[1]]).all().compute()

    orig = op.join(result)
    assert (orig == example_data).all().compute()
