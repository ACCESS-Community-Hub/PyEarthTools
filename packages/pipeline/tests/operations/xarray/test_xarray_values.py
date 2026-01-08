# Copyright Commonwealth of Australia, Bureau of Meteorology 2025.
#
# Licensed under the Apache License, Version 2 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pyearthtools.pipeline.operations.xarray import values

import numpy as np
import xarray as xr
import pytest


@pytest.fixture(scope="module")
def example_dataarray():
    return xr.DataArray(
        [
            [1, 2, 3],
            [-np.inf, np.inf, -np.inf],
            [1, np.nan, np.nan],
        ]
    )


@pytest.fixture(scope="module")
def example_dataset(example_dataarray):
    return xr.Dataset(
        {
            "a": example_dataarray,
            "b": 2 * example_dataarray,
        }
    )


def test_fillnan(example_dataarray, example_dataset):
    """Tests xarray FillNan operation class for DataArray and Dataset inputs."""
    op = values.FillNan(nan=123, posinf=456, neginf=-789)
    result = op.apply_func(example_dataarray)

    assert result.equals(
        xr.DataArray(
            [
                [1, 2, 3],
                [-789, 456, -789],
                [1, 123, 123],
            ],
        )
    )

    result = op.apply_func(example_dataset)

    assert result.equals(
        xr.Dataset(
            {
                "a": xr.DataArray(
                    [
                        [1, 2, 3],
                        [-789, 456, -789],
                        [1, 123, 123],
                    ],
                ),
                "b": xr.DataArray(
                    [
                        [2, 4, 6],
                        [-789, 456, -789],
                        [2, 123, 123],
                    ],
                ),
            }
        )
    )

    with pytest.raises(TypeError):
        op.apply_func(1)


def test_maskvalue(example_dataarray, example_dataset):
    """Tests xarray MaskValue operation class."""

    # pass invalid operation
    with pytest.raises(KeyError):
        values.MaskValue(1, operation="*")

    # test default op (==)
    op = values.MaskValue(1)
    result = op.apply_func(example_dataarray)

    assert result.equals(
        xr.DataArray(
            [
                [np.nan, 2, 3],
                [-np.inf, np.inf, -np.inf],
                [np.nan, np.nan, np.nan],
            ]
        )
    )

    result = op.apply_func(example_dataset)

    assert result.equals(
        xr.Dataset(
            {
                "a": xr.DataArray(
                    [
                        [np.nan, 2, 3],
                        [-np.inf, np.inf, -np.inf],
                        [np.nan, np.nan, np.nan],
                    ]
                ),
                "b": xr.DataArray(
                    [
                        [2, 4, 6],
                        [-np.inf, np.inf, -np.inf],
                        [2, np.nan, np.nan],
                    ]
                ),
            }
        )
    )

    # test <= op
    op = values.MaskValue(2, operation="<=")
    result = op.apply_func(example_dataarray)

    assert result.equals(
        xr.DataArray(
            [
                [np.nan, np.nan, 3],
                [np.nan, np.inf, np.nan],
                [np.nan, np.nan, np.nan],
            ]
        )
    )

    result = op.apply_func(example_dataset)

    assert result.equals(
        xr.Dataset(
            {
                "a": xr.DataArray(
                    [
                        [np.nan, np.nan, 3],
                        [np.nan, np.inf, np.nan],
                        [np.nan, np.nan, np.nan],
                    ]
                ),
                "b": xr.DataArray([[np.nan, 4, 6], [np.nan, np.inf, np.nan], [np.nan, np.nan, np.nan]]),
            }
        )
    )

    # test < op
    op = values.MaskValue(2, operation="<")
    result = op.apply_func(example_dataarray)

    assert result.equals(
        xr.DataArray(
            [
                [np.nan, 2, 3],
                [np.nan, np.inf, np.nan],
                [np.nan, np.nan, np.nan],
            ]
        )
    )

    result = op.apply_func(example_dataset)

    assert result.equals(
        xr.Dataset(
            {
                "a": xr.DataArray(
                    [
                        [np.nan, 2, 3],
                        [np.nan, np.inf, np.nan],
                        [np.nan, np.nan, np.nan],
                    ]
                ),
                "b": xr.DataArray([[2, 4, 6], [np.nan, np.inf, np.nan], [2, np.nan, np.nan]]),
            }
        )
    )

    # test >= op
    op = values.MaskValue(2, operation=">=")
    result = op.apply_func(example_dataarray)

    assert result.equals(
        xr.DataArray(
            [
                [1, np.nan, np.nan],
                [-np.inf, np.nan, -np.inf],
                [1, np.nan, np.nan],
            ]
        )
    )

    result = op.apply_func(example_dataset)

    assert result.equals(
        xr.Dataset(
            {
                "a": xr.DataArray(
                    [
                        [1, np.nan, np.nan],
                        [-np.inf, np.nan, -np.inf],
                        [1, np.nan, np.nan],
                    ],
                ),
                "b": xr.DataArray(
                    [
                        [np.nan, np.nan, np.nan],
                        [-np.inf, np.nan, -np.inf],
                        [np.nan, np.nan, np.nan],
                    ]
                ),
            }
        )
    )

    # test > op
    op = values.MaskValue(2, operation=">")
    result = op.apply_func(example_dataarray)

    assert result.equals(
        xr.DataArray(
            [
                [1, 2, np.nan],
                [-np.inf, np.nan, -np.inf],
                [1, np.nan, np.nan],
            ],
        ),
    )

    result = op.apply_func(example_dataset)

    assert result.equals(
        xr.Dataset(
            {
                "a": xr.DataArray(
                    [
                        [1, 2, np.nan],
                        [-np.inf, np.nan, -np.inf],
                        [1, np.nan, np.nan],
                    ],
                ),
                "b": xr.DataArray(
                    [
                        [2, np.nan, np.nan],
                        [-np.inf, np.nan, -np.inf],
                        [2, np.nan, np.nan],
                    ]
                ),
            }
        )
    )
