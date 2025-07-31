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

from pyearthtools.data.transforms import coordinates
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

SIMPLE_DA2 = xr.DataArray(
    [
        [0.9, 0.0, 5],
        [0.7, 1.4, 2.8],
        [0.4, 0.5, 2.3],
    ],
    coords=[[0, 1, 2], [5, 6, 7]],
    dims=["lat", "lon"],
)

SIMPLE_DS1 = xr.Dataset({"Temperature": SIMPLE_DA1})
SIMPLE_DS2 = xr.Dataset({"Humidity": SIMPLE_DA1, "Temperature": SIMPLE_DA1, "WombatsPerKm2": SIMPLE_DA1})

COMPLICATED_DS1 = xr.Dataset({"Temperature": SIMPLE_DA1, "MSLP": SIMPLE_DA2})


def test_Flatten():
    f = coordinates.Flatten(["height"])
    output = f.apply(SIMPLE_DS2)
    variables = list(output.keys())
    for vbl in ["Temperature10", "Temperature20", "Humidity10", "Humidity20", "WombatsPerKm210", "WombatsPerKm220"]:
        assert vbl in variables


def test_Flatten_2_coords():
    f = coordinates.Flatten(["height", "lon"])
    output = f.apply(SIMPLE_DS1)
    variables = list(output.keys())
    # Note that it's hard to predict which coordinate will be processed first.
    try:
        for vbl in [
            "Temperature510",
            "Temperature520",
            "Temperature610",
            "Temperature620",
            "Temperature710",
            "Temperature720",
        ]:
            assert vbl in variables
    except AssertionError:
        for vbl in [
            "Temperature105",
            "Temperature205",
            "Temperature106",
            "Temperature206",
            "Temperature107",
            "Temperature207",
        ]:
            assert vbl in variables


def test_Flatten_complicated_dataset():
    """Check that Flatten still works when the coordinate being flattened does not exist for all variables."""
    f = coordinates.Flatten(["height"])
    output = f.apply(COMPLICATED_DS1)
    variables = list(output.keys())
    for vbl in ["Temperature10", "Temperature20", "MSLP"]:
        assert vbl in variables


def test_Flatten_skip_missing():
    f = coordinates.Flatten(["scrupulosity"])
    with pytest.raises(ValueError):
        f.apply(SIMPLE_DS1)
    f2 = coordinates.Flatten(["scrupulosity"], skip_missing=True)
    output2 = f2.apply(SIMPLE_DS1)
    assert output2 == SIMPLE_DS1, "When skip_missing=True, Datasets without the given coordinate pass unchanged."
