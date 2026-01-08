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

from pyearthtools.pipeline.operations.dask import values

import dask.array as da
import numpy as np
import pytest


@pytest.fixture(scope="module")
def example_data():
    return da.array(
        [
            [1, 2, 3],
            [-np.inf, np.inf, -np.inf],
            [1, np.nan, np.nan],
        ]
    )


def test_fillnan(example_data):
    """Tests dask FillNan operation class."""
    op = values.FillNan(nan=123, posinf=456, neginf=-789)
    result = op.apply_func(example_data)

    assert (
        (
            result
            == da.array(
                [
                    [1, 2, 3],
                    [-789, 456, -789],
                    [1, 123, 123],
                ]
            )
        )
        .all()
        .compute()
    )
