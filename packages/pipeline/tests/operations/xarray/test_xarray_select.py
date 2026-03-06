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

import xarray as xr
import pytest
import numpy as np


from pyearthtools.pipeline.operations.xarray import select


@pytest.fixture(scope="module")
def sample():
    """Test xarray dataset."""
    coords = {"dim0": range(3), "dim1": range(3)}
    return xr.Dataset(
        {
            "var1": xr.DataArray(np.array(range(9)).reshape((3, 3)), coords),
            "var2": xr.DataArray(np.array(range(9, 18)).reshape((3, 3)), coords),
        },
    )


def test_SelectDataset(sample):
    """Tests the SelectDataset xarray operation."""

    s = select.SelectDataset(("var1",))

    output = s.apply_func(sample)

    assert "var1" in output
    assert "var2" not in output
    assert output["var1"].equals(sample["var1"])


def test_DropDataset(sample):
    """Tests the DropDataset xarray operation."""

    s = select.DropDataset(("var1",))

    output = s.apply_func(sample)
    assert "var1" not in output
    assert "var2" in output
    assert output["var2"].equals(sample["var2"])


def test_SliceDataset(sample):
    """Tests the SliceDataset xarray operation."""

    args = {"dim0": (0, 2, 2), "dim1": (0, 1)}

    def test_slicer(slicer, sample):

        output = s.apply_func(sample)

        assert np.array_equal(output.coords["dim0"].values, [0, 2])
        assert np.array_equal(output.coords["dim1"].values, [0, 1])

    # test passing dict to SliceDataset
    s = select.SliceDataset(args)
    test_slicer(s, sample)

    # test passing kwargs to SliceDataset
    s = select.SliceDataset(**args)
    test_slicer(s, sample)

    # test passing dataarray to slicer
    test_slicer(s, sample["var1"])
