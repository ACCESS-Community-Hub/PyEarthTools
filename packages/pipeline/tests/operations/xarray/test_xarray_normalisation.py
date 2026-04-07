# Copyright Commonwealth of Australia, Bureau of Meteorology 2026.
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

from pyearthtools.pipeline.operations.xarray.normalisation import Anomaly, Deviation, Division, Evaluated

import numpy as np
import xarray as xr
import pytest


def _write_data(file_path, data, dims=None):
    "Helper function to save numpy data as an xarray Dataset"
    if dims is None:
        if isinstance(data, np.ndarray) and data.ndim == 2:
            dims = ["x", "y"]
        elif isinstance(data, np.ndarray) and data.ndim == 1:
            dims = ["x"]  # Default, might need to be overridden by tests
        else:
            dims = []

    da = xr.DataArray(data, dims=dims, name="data")
    ds = xr.Dataset({"data": da})
    ds.to_netcdf(file_path)


@pytest.fixture(params=["dataarray", "dataset"])
def sample(request):
    "A simple sequential xarray object, parametrized as DataArray and Dataset."
    data = np.array(range(6)).reshape(3, 2)
    da = xr.DataArray(data, dims=["x", "y"], name="data")
    if request.param == "dataarray":
        return da
    return xr.Dataset({"data": da})


@pytest.mark.parametrize(
    ("mean_data", "dims", "expected"),
    (
        (np.array(range(3)), ["x"], [[0, 1], [1, 2], [2, 3]]),  # mean broadcasted over dim=0
        (np.array(range(2)), ["y"], [[0, 0], [2, 2], [4, 4]]),  # mean broadcasted over dim=-1
        (np.array(range(6, 12)).reshape(3, 2), ["x", "y"], np.repeat(-6, 6).reshape(3, 2)),
    ),
)
def test_Anomaly(sample, tmp_path, mean_data, dims, expected):
    "Tests the Anomaly normalisation class."

    file_path = tmp_path / "mean.nc"
    _write_data(file_path=file_path, data=mean_data, dims=dims)

    a = Anomaly(mean=file_path)

    result = a.apply_func(sample)

    if isinstance(sample, xr.DataArray) and isinstance(result, xr.Dataset):
        expected_xr = sample.to_dataset(name=sample.name or "data").copy(data={sample.name or "data": expected})
    elif isinstance(sample, xr.DataArray):
        expected_xr = sample.copy(data=expected)
    else:
        expected_xr = sample.copy(deep=True)
        expected_xr["data"] = (["x", "y"], expected)

    xr.testing.assert_allclose(result, expected_xr)

    result_reversed = a.undo_func(result)
    if isinstance(sample, xr.DataArray) and isinstance(result_reversed, xr.Dataset):
        expected_reversed = sample.to_dataset(name=sample.name or "data")
    else:
        expected_reversed = sample
    xr.testing.assert_allclose(result_reversed, expected_reversed)


@pytest.mark.parametrize(
    ("mean_data", "dims", "expected"),
    (
        (np.array(range(3)), ["x"], [[0, 10], [5, 10], [20 / 3, 10]]),  # mean broadcasted over dim=0
        (np.array(range(2)), ["y"], [[0, 0], [20, 10], [40, 20]]),  # mean broadcasted over dim=-1
        (np.array(range(6, 12)).reshape(3, 2), ["x", "y"], -6 / (0.1 * np.array(range(7, 13)).reshape(3, 2))),
    ),
)
def test_Deviation(sample, tmp_path, mean_data, dims, expected):
    "Tests the Deviation normalisation class."

    mean_file_path = tmp_path / "mean.nc"
    stdev_file_path = tmp_path / "stdev.nc"
    _write_data(file_path=mean_file_path, data=mean_data, dims=dims)
    _write_data(file_path=stdev_file_path, data=0.1 * (mean_data + 1), dims=dims)  # +1 to avoid divide by 0

    a = Deviation(mean=mean_file_path, deviation=stdev_file_path)

    result = a.apply_func(sample)

    if isinstance(sample, xr.DataArray) and isinstance(result, xr.Dataset):
        expected_xr = sample.to_dataset(name=sample.name or "data").copy(data={sample.name or "data": expected})
    elif isinstance(sample, xr.DataArray):
        expected_xr = sample.copy(data=expected)
    else:
        expected_xr = sample.copy(deep=True)
        expected_xr["data"] = (["x", "y"], expected)

    xr.testing.assert_allclose(result, expected_xr)

    result_reversed = a.undo_func(result)
    if isinstance(sample, xr.DataArray) and isinstance(result_reversed, xr.Dataset):
        expected_reversed = sample.to_dataset(name=sample.name or "data")
    else:
        expected_reversed = sample
    xr.testing.assert_allclose(result_reversed, expected_reversed)


@pytest.mark.parametrize(
    ("div_data", "dims", "expected"),
    (
        (np.array(range(1, 4)), ["x"], [[0, 1], [1, 1.5], [4 / 3, 5 / 3]]),  # mean broadcasted over dim=0
        (np.array(range(1, 3)), ["y"], [[0, 0.5], [2, 1.5], [4, 2.5]]),  # mean broadcasted over dim=-1
        (np.array(range(6, 12)).reshape(3, 2), ["x", "y"], [[0, 1 / 7], [2 / 8, 1 / 3], [2 / 5, 5 / 11]]),
    ),
)
def test_Division(sample, tmp_path, div_data, dims, expected):
    "Tests the Division normalisation class."

    file_path = tmp_path / "div.nc"
    _write_data(file_path=file_path, data=div_data, dims=dims)

    a = Division(division_factor=file_path)

    result = a.apply_func(sample)

    if isinstance(sample, xr.DataArray) and isinstance(result, xr.Dataset):
        expected_xr = sample.to_dataset(name=sample.name or "data").copy(data={sample.name or "data": expected})
    elif isinstance(sample, xr.DataArray):
        expected_xr = sample.copy(data=expected)
    else:
        expected_xr = sample.copy(deep=True)
        expected_xr["data"] = (["x", "y"], expected)

    xr.testing.assert_allclose(result, expected_xr)

    result_reversed = a.undo_func(result)
    if isinstance(sample, xr.DataArray) and isinstance(result_reversed, xr.Dataset):
        expected_reversed = sample.to_dataset(name=sample.name or "data")
    else:
        expected_reversed = sample
    xr.testing.assert_allclose(result_reversed, expected_reversed)


@pytest.mark.parametrize(
    ("norm_str", "denorm_str", "kwargs", "expected"),
    (
        ("sample + 1", "sample - 1", {}, np.array(range(1, 7)).reshape(3, 2)),
        ("sample + var", "sample - var", {"var": 5}, np.array(range(5, 11)).reshape(3, 2)),
        ("sample + var", "sample - var", {"var": "file"}, np.array([[0, 2], [2, 4], [4, 6]])),
    ),
)
def test_Evaluated(sample, tmp_path, norm_str, denorm_str, kwargs, expected):
    "Tests the Evaluated normalisation class."

    if kwargs.get("var") == "file":
        file_path = tmp_path / "some_vals.nc"
        _write_data(file_path, np.array(range(2)), dims=["y"])
        kwargs["var"] = file_path

    e = Evaluated(normalisation_eval=norm_str, unnormalisation_eval=denorm_str, **kwargs)

    result = e.apply_func(sample)

    if isinstance(sample, xr.DataArray) and isinstance(result, xr.Dataset):
        expected_xr = sample.to_dataset(name=sample.name or "data").copy(data={sample.name or "data": expected})
    elif isinstance(sample, xr.DataArray):
        expected_xr = sample.copy(data=expected)
    else:
        expected_xr = sample.copy(deep=True)
        expected_xr["data"] = (["x", "y"], expected)

    xr.testing.assert_allclose(result, expected_xr)

    result_reverse = e.undo_func(result)
    if isinstance(sample, xr.DataArray) and isinstance(result_reverse, xr.Dataset):
        expected_reversed = sample.to_dataset(name=sample.name or "data")
    else:
        expected_reversed = sample
    xr.testing.assert_allclose(result_reverse, expected_reversed)
