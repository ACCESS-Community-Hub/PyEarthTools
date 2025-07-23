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

from pyearthtools.pipeline.operations.xarray import reshape

import xarray as xr
import pytest

SIMPLE_DA1 = xr.DataArray(
    [
        [
            [0.9, 0.0, 5],
            [0.7, 1.4, 2.8],
            [0.4, 0.5, 2.3],
        ],
        [
            [1.9, 1.0, 1.5],
            [1.7, 2.4, 1.1],
            [1.4, 1.5, 3.3],
        ],
    ],
    coords=[[10, 20], [0, 1, 2], [5, 6, 7]],
    dims=["height", "lat", "lon"],
)
SIMPLE_DS1 = xr.Dataset({"Temperature": SIMPLE_DA1})
SIMPLE_DS2 = xr.Dataset({"Humidity": SIMPLE_DA1, "Temperature": SIMPLE_DA1, "WombatsPerKm2": SIMPLE_DA1})

def test_Dimensions():
    d = reshape.Dimensions(["lat", "lon", "height"])
    output = d.apply_func(SIMPLE_DA1)
    assert output.dims == ("lat", "lon", "height")

def test_Dimensions_one_input():
    d = reshape.Dimensions(["lat"])
    output = d.apply_func(SIMPLE_DA1)
    assert output.dims[0] == "lat"

def test_Dimensions_prepend():
    d = reshape.Dimensions(["lat"], append=False)
    output = d.apply_func(SIMPLE_DA1)
    assert output.dims[-1] == "lat"

def test_Dimensions_preserve_order():
    d = reshape.Dimensions(["lat"], preserve_order=True)
    output = d.apply_func(SIMPLE_DA1)
    reversed_output = d.undo_func(output)
    assert reversed_output.dims == output.dims


def test_CoordinateFlatten():
    cf = reshape.CoordinateFlatten(coordinate=("height", (10, 20)), skip_missing=True)
    cf.apply_func(SIMPLE_DA1)





    # def test_Flatten():
    #     f1 = reshape.Flatten(flatten_dims=2)
    #     random_array = np.random.randn(4, 3, 5)
    #     output = f1.apply_func(random_array)
    #     undo_output = f1.undo_func(output)
    #     assert output.shape == (4, 3 * 5), "Flatten acts on the last few dimensions."
    #     assert np.all(undo_output == random_array), "Flatten can undo itself."
    #
    # def test_Flatten_1_dim():
    #     f2 = reshape.Flatten(flatten_dims=1)
    #     random_array = np.random.randn(4, 3, 5)
    #     output = f2.apply_func(random_array)
    #     undo_output = f2.undo_func(output)  # Check that the undo still works.
    #     assert np.all(output == random_array), "Flatten 1 dimension does nothing."
    #     assert np.all(undo_output == random_array), "Undo Flatten 1 dimension."
    #
    # def test_Flatten_all_dims():
    #     f3 = reshape.Flatten()
    #     random_array3 = np.random.randn(6, 7, 5, 2)
    #     output = f3.apply_func(random_array3)
    #     assert output.shape == (6 * 7 * 5 * 2,)
    #     assert f3.undo_func(output).shape == (6, 7, 5, 2), "Undo Flatten all dimensions."


