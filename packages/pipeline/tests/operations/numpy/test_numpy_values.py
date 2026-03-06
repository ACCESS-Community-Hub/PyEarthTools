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

from pyearthtools.pipeline.operations.numpy import values

import numpy as np
import pytest


@pytest.fixture(scope="module")
def example_data():
    return np.array(
        [
            [1, 2, 3],
            [-np.inf, np.inf, -np.inf],
            [1, np.nan, np.nan],
        ]
    )


def test_fillnan(example_data):
    """Tests numpy FillNan operation class."""
    op = values.FillNan(nan=123, posinf=456, neginf=-789)
    result = op.apply_func(example_data)

    assert np.array_equal(
        result,
        np.array(
            [
                [1, 2, 3],
                [-789, 456, -789],
                [1, 123, 123],
            ],
            dtype=result.dtype,
        ),
    )


def test_maskvalue(example_data):
    """Tests numpy MaskValue operation class."""

    # pass invalid operation
    with pytest.raises(KeyError):
        values.MaskValue(1, operation="*")

    # test default op (==)
    op = values.MaskValue(1)
    result = op.apply_func(example_data)

    assert np.array_equal(
        result,
        np.array(
            [
                [np.nan, 2.0, 3.0],
                [-np.inf, np.inf, -np.inf],
                [np.nan, np.nan, np.nan],
            ],
            dtype=result.dtype,
        ),
        equal_nan=True,
    )

    # test <= op
    op = values.MaskValue(2, operation="<=")
    result = op.apply_func(example_data)

    assert np.array_equal(
        result,
        np.array(
            [
                [np.nan, np.nan, 3.0],
                [np.nan, np.inf, np.nan],
                [np.nan, np.nan, np.nan],
            ],
            dtype=result.dtype,
        ),
        equal_nan=True,
    )

    # test < op
    op = values.MaskValue(2, operation="<")
    result = op.apply_func(example_data)

    assert np.array_equal(
        result,
        np.array(
            [
                [np.nan, 2.0, 3.0],
                [np.nan, np.inf, np.nan],
                [np.nan, np.nan, np.nan],
            ],
            dtype=result.dtype,
        ),
        equal_nan=True,
    )

    # test >= op
    op = values.MaskValue(2, operation=">=")
    result = op.apply_func(example_data)

    assert np.array_equal(
        result,
        np.array(
            [
                [1.0, np.nan, np.nan],
                [-np.inf, np.nan, -np.inf],
                [1.0, np.nan, np.nan],
            ],
            dtype=result.dtype,
        ),
        equal_nan=True,
    )

    # test > op
    op = values.MaskValue(2, operation=">")
    result = op.apply_func(example_data)

    assert np.array_equal(
        result,
        np.array(
            [
                [1.0, 2.0, np.nan],
                [-np.inf, np.nan, -np.inf],
                [1.0, np.nan, np.nan],
            ],
            dtype=result.dtype,
        ),
        equal_nan=True,
    )


def test_clip(example_data):
    """Tests numpy Clip operation class."""
    op = values.Clip()
    result = op.apply_func(example_data)

    assert np.array_equal(
        result,
        np.array(
            [
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, np.nan, np.nan],
            ],
            dtype=result.dtype,
        ),
        equal_nan=True,
    )
