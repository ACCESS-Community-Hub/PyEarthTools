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

from pyearthtools.pipeline.operations.xarray import split

import numpy as np
import xarray as xr
import pytest


@pytest.fixture(scope="module")
def example_dataarray():
    return xr.DataArray(
        np.array(range(2 * 3 * 4)).reshape((2, 3, 4)), coords={"c0": range(2), "c1": range(3), "c2": range(4)}
    )


@pytest.fixture(scope="module")
def example_dataset(example_dataarray):
    return xr.Dataset({"a": example_dataarray, "b": 2 * example_dataarray})


def test_onvariables(example_dataset):
    """Tests xarray OnVariables operation class."""

    # split on all variables
    op = split.OnVariables()
    result = op.split(example_dataset)
    assert result[0].equals(example_dataset.drop_vars("b"))
    assert result[1].equals(example_dataset.drop_vars("a"))

    # join datasets
    orig = op.join(result)
    assert orig.equals(example_dataset)

    # split on selected variables
    op = split.OnVariables(variables=("a",))
    result = op.split(example_dataset)
    assert result[0].equals(example_dataset.drop_vars("b"))

    # split on non-existent variable
    op = split.OnVariables(variables=("c",))
    with pytest.raises(ValueError):
        op.split(example_dataset)


def test_oncoordinate(example_dataarray, example_dataset):
    """Tests xarray OnCoordinate operation class."""
    op = split.OnCoordinate("c1")
    result = op.split(example_dataset)
    for i, arr in enumerate(result):
        assert arr["a"].equals(example_dataset["a"].loc[:, i, :])
        assert arr["b"].equals(example_dataset["b"].loc[:, i, :])

    orig = op.join(result)
    assert orig.broadcast_equals(example_dataset)

    result = op.split(example_dataarray)
    for i, arr in enumerate(result):
        assert arr.equals(example_dataarray.loc[:, i, :])

    orig = op.join(result)
    assert orig.broadcast_equals(example_dataarray)
