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

from pyearthtools.pipeline.operations.xarray.normalisation import (
    Anomaly,
    Deviation,
    Division,
    SingleValueDivision,
    Evaluated,
    MagicNorm,
)

import os
import numpy as np
import xarray as xr
import pytest


def _write_data(file_path, data, dims=None):
    "Helper function to save numpy data as an xarray Dataset"

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
@pytest.mark.parametrize("input_type", ["file", "dataset", "dataarray"])
def test_Deviation(sample, tmp_path, mean_data, dims, expected, input_type):
    "Tests the Deviation normalisation class."

    mean_file_path = tmp_path / "mean.nc"
    stdev_file_path = tmp_path / "stdev.nc"
    _write_data(file_path=mean_file_path, data=mean_data, dims=dims)
    _write_data(file_path=stdev_file_path, data=0.1 * (mean_data + 1), dims=dims)  # +1 to avoid divide by 0

    if input_type == "file":
        mean_val = mean_file_path
        stdev_val = stdev_file_path
    elif input_type == "dataset":
        mean_val = xr.open_dataset(mean_file_path)
        stdev_val = xr.open_dataset(stdev_file_path)
    elif input_type == "dataarray":
        mean_val = xr.open_dataset(mean_file_path)["data"]
        stdev_val = xr.open_dataset(stdev_file_path)["data"]

    a = Deviation(mean=mean_val, deviation=stdev_val)

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


def test_Deviation_float(sample):
    "Tests the Deviation normalisation class with float inputs."
    mean_val = 2.0
    stdev_val = 0.5
    expected = (np.array(range(6)).reshape(3, 2) - mean_val) / stdev_val

    a = Deviation(mean=mean_val, deviation=stdev_val)

    result = a.apply_func(sample)

    if isinstance(sample, xr.DataArray):
        expected_xr = sample.copy(data=expected)
    else:
        expected_xr = sample.copy(deep=True)
        expected_xr["data"] = (["x", "y"], expected)

    xr.testing.assert_allclose(result, expected_xr)

    result_reversed = a.undo_func(result)
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


def test_SingleValueDivision(sample):
    "Tests the SingleValueDivision normalisation class."
    div_val = 2.0
    expected = np.array(range(6)).reshape(3, 2) / div_val
    a = SingleValueDivision(division_factor=div_val)

    result = a.apply_func(sample)

    if isinstance(sample, xr.DataArray):
        expected_xr = sample.copy(data=expected)
    else:
        expected_xr = sample.copy(deep=True)
        expected_xr["data"] = (["x", "y"], expected)

    xr.testing.assert_allclose(result, expected_xr)

    result_reversed = a.undo_func(result)
    expected_reversed = sample
    xr.testing.assert_allclose(result_reversed, expected_reversed)


def test_MagicNorm(sample, tmp_path):
    "Tests the MagicNorm normalisation class."
    # Use 2 samples to have a non-zero std
    samples_needed = 2
    m = MagicNorm(cache_dir=tmp_path, samples_needed=samples_needed)

    # First sample
    s1 = sample
    # Second sample (different values to avoid 0 std)
    s2 = sample + 10

    # Pass enough samples to fix the norms
    _ = m.apply_func(s1)
    _ = m.apply_func(s2)

    # Verify files created
    assert os.path.exists(tmp_path / "magic_means.nc")
    assert os.path.exists(tmp_path / "magic_std.nc")

    # Verify loaded in next instance
    m2 = MagicNorm(cache_dir=tmp_path, samples_needed=samples_needed)
    assert m2.mean is not None
    assert m2.deviation is not None

    # Calculate expected
    combined = xr.concat([s1, s2], dim="samples")
    expected_mean = combined.mean()
    expected_std = combined.std()

    # m2.mean is always a Dataset
    if isinstance(sample, xr.DataArray):
        xr.testing.assert_allclose(m2.mean["data"], expected_mean)
        xr.testing.assert_allclose(m2.deviation["data"], expected_std)
    else:
        xr.testing.assert_allclose(m2.mean, expected_mean)
        xr.testing.assert_allclose(m2.deviation, expected_std)

    # Now verify identity with fixed norms
    res1_final = m.apply_func(s1)
    res1_undo = m.undo_func(res1_final)

    expected_reversed = sample
    xr.testing.assert_allclose(res1_undo, expected_reversed)


def test_MagicNorm_caching(sample, tmp_path):
    "Tests the caching behavior of MagicNorm."
    samples_needed = 2
    m = MagicNorm(cache_dir=tmp_path, samples_needed=samples_needed)

    # Pass enough samples to trigger caching
    _ = m.apply_func(sample)
    _ = m.apply_func(sample + 10)

    # Verify files created
    assert os.path.exists(tmp_path / "magic_means.nc")
    assert os.path.exists(tmp_path / "magic_std.nc")

    # Initialize a NEW instance with the same cache_dir
    m_new = MagicNorm(cache_dir=tmp_path, samples_needed=100)  # samples_needed should be ignored because files exist

    # Verify values loaded
    assert m_new.mean is not None
    assert m_new.deviation is not None

    if isinstance(sample, xr.DataArray):
        xr.testing.assert_allclose(m_new.mean["data"], m.mean)
        xr.testing.assert_allclose(m_new.deviation["data"], m.deviation)
    else:
        xr.testing.assert_allclose(m_new.mean, m.mean)
        xr.testing.assert_allclose(m_new.deviation, m.deviation)

    # Verify it can normalise using the cached values
    res = m_new.apply_func(sample)
    res_orig = m.apply_func(sample)

    if isinstance(sample, xr.DataArray) and isinstance(res, xr.Dataset):
        res_orig_cmp = res_orig.to_dataset(name=sample.name or "data")
    else:
        res_orig_cmp = res_orig
    xr.testing.assert_allclose(res, res_orig_cmp)


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


def test_MagicNorm_early_return(sample, tmp_path):
    "Tests the early return branch (line 116) in MagicNorm.update_norms."
    m = MagicNorm(cache_dir=tmp_path, samples_needed=1)

    # First sample - calculates norms
    _ = m.apply_func(sample)

    # Assert sample_count is exactly 1
    assert m.sample_count == 1

    # Call update_norms explicitly with a second sample
    m.update_norms(sample + 10)

    # sample_count should still be 1 because update_norms returns early
    assert m.sample_count == 1


def test_MagicNorm_multithreading_simulation(sample, tmp_path):
    "Tests the os.path.exists branches (lines 121 and 135) in MagicNorm.update_norms simulating multithreading."
    m1 = MagicNorm(cache_dir=tmp_path, samples_needed=2)

    # Pass 1 sample to m1
    _ = m1.apply_func(sample)
    assert m1.sample_count == 1

    # Initialize m2 which will reach the cache threshold and write files
    m2 = MagicNorm(cache_dir=tmp_path, samples_needed=2)
    _ = m2.apply_func(sample)
    _ = m2.apply_func(sample + 10)

    # The cache files are now created by m2
    assert os.path.exists(tmp_path / "magic_means.nc")

    # Now call apply_func on m1 with a new sample. This calls m1.update_norms(sample + 20).
    # Inside m1.update_norms, it will find the files and load them instead of appending and recalculating.
    _ = m1.apply_func(sample + 20)

    # Verify m1 loaded the cache
    assert m1.mean is not None
    assert m1.deviation is not None

    if isinstance(sample, xr.DataArray):
        xr.testing.assert_allclose(m1.mean["data"], m2.mean)
        xr.testing.assert_allclose(m1.deviation["data"], m2.deviation)
    else:
        xr.testing.assert_allclose(m1.mean, m2.mean)
        xr.testing.assert_allclose(m1.deviation, m2.deviation)
