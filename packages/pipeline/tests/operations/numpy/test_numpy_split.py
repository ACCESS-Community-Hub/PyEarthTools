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

from pyearthtools.pipeline.operations.numpy import split

import numpy as np
import pytest


@pytest.fixture(scope="module")
def example_data():
    return np.array(range(2 * 3 * 4)).reshape((2, 3, 4))


def test_onaxis(example_data):
    """Tests numpy OnAxis split operation class."""
    op = split.OnAxis(axis=1)

    # try join before splitting
    with pytest.raises(RuntimeError):
        op.join((example_data, example_data))
    result = op.split(example_data)
    assert all(np.array_equal(arr, example_data[:, d, :]) for d, arr in enumerate(result))

    orig = op.join(result)
    assert np.array_equal(orig, example_data)
