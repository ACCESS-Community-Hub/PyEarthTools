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

SIMPLE_DA2 = xr.DataArray([[9.1, 2.3, 3.2], [2.2, 1.1, 0.2]], coords=[[1, 2], [3, 4, 5]], dims=["a", "b"])

SIMPLE_DS1 = xr.Dataset({"Temperature": SIMPLE_DA1})
SIMPLE_DS2 = xr.Dataset({"Humidity": SIMPLE_DA1, "Temperature": SIMPLE_DA1, "WombatsPerKm2": SIMPLE_DA1})

COMPLICATED_DS1 = xr.Dataset({"Temperature": SIMPLE_DA1, "MSLP": SIMPLE_DA2})


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


def test_weak_cast_to_int():

    wcti = reshape.weak_cast_to_int

    assert wcti(5.0) == 5
    assert isinstance(wcti(5.0), int)

    assert wcti("hello") == "hello"


def test_CoordinateFlatten():
    f = reshape.CoordinateFlatten(["height"])
    output = f.apply(SIMPLE_DS2)
    variables = list(output.keys())
    for vbl in ["Temperature10", "Temperature20", "Humidity10", "Humidity20", "WombatsPerKm210", "WombatsPerKm220"]:
        assert vbl in variables


def test_CoordinateFlatten_complicated_dataset():
    """Check that Flatten still works when the coordinate being flattened does not exist for all variables."""
    f = reshape.CoordinateFlatten(["height"])
    output = f.apply(COMPLICATED_DS1)
    variables = list(output.keys())
    for vbl in ["Temperature10", "Temperature20", "MSLP"]:
        assert vbl in variables


def test_CoordinateFlatten_skip_missing():
    f = reshape.CoordinateFlatten(["scrupulosity"])
    with pytest.raises(ValueError):
        f.apply(SIMPLE_DS1)
    f2 = reshape.CoordinateFlatten(["scrupulosity"], skip_missing=True)
    output2 = f2.apply(SIMPLE_DS1)
    assert output2 == SIMPLE_DS1, "When skip_missing=True, Datasets without the given coordinate pass unchanged."


def test_undo_CoordinateFlatten():
    f = reshape.CoordinateFlatten(["height"])
    f_output = f.apply(SIMPLE_DS2)
    f_undone = f.undo(f_output)
    variables = list(f_undone.keys())
    for vbl in ["Temperature", "Humidity", "WombatsPerKm2"]:
        assert vbl in variables


def test_CoordinateExpand_reverses_CoordinateFlatten():
    f = reshape.CoordinateFlatten(["height"])
    f_output = f.apply(SIMPLE_DS2)
    e = reshape.CoordinateExpand(["height"])
    e_output = e.apply(f_output)
    variables = list(e_output.keys())
    assert "Temperature" in variables


def test_undo_CoordinateExpand():
    f = reshape.CoordinateFlatten(["height"])
    f_output = f.apply(SIMPLE_DS2)
    e = reshape.CoordinateExpand(["height"])
    e_output = e.apply(f_output)
    e_undone = e.undo(e_output)
    variables = list(e_undone.keys())
    for vbl in ["Temperature10", "Temperature20", "Humidity10", "Humidity20", "WombatsPerKm210", "WombatsPerKm220"]:
        assert vbl in variables
