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

from pyearthtools.pipeline.operations.dask.normalisation import Anomaly, Deviation, Division, Evaluated

import numpy as np
import dask.array as da
import pytest


def _write_data(file_path, data):
    "Helper function to save numpy data"
    np.save(file_path, data)


@pytest.fixture
def sample():
    "A simple sequential dask array"
    return da.from_array(np.array(range(6)).reshape(3, 2))


@pytest.mark.parametrize(
    ("mean_data", "expand", "expected"),
    (
        (np.array(range(3)), True, [[0, 1], [1, 2], [2, 3]]),  # mean broadcasted over dim=0
        (np.array(range(2)), False, [[0, 0], [2, 2], [4, 4]]),  # mean broadcasted over dim=-1
        (np.array(range(6, 12)).reshape(3, 2), True, np.repeat(-6, 6).reshape(3, 2)),
    ),
)
def test_Anomaly(sample, tmp_path, mean_data, expand, expected):
    "Tests the Anomaly normalisation class for dask."

    file_path = tmp_path / "mean.npy"
    _write_data(file_path=file_path, data=mean_data)

    a = Anomaly(mean=file_path, expand=expand)

    result = a.apply_func(sample)
    np.testing.assert_array_equal(result.compute(), np.array(expected))

    result_reversed = a.undo_func(result)
    np.testing.assert_array_equal(result_reversed.compute(), sample.compute())


@pytest.mark.parametrize(
    ("mean_data", "expand", "expected"),
    (
        (np.array(range(3)), True, [[0, 10], [5, 10], [20 / 3, 10]]),  # mean broadcasted over dim=0
        (np.array(range(2)), False, [[0, 0], [20, 10], [40, 20]]),  # mean broadcasted over dim=-1
        (np.array(range(6, 12)).reshape(3, 2), True, -6 / (0.1 * np.array(range(7, 13)).reshape(3, 2))),
    ),
)
def test_Deviation(sample, tmp_path, mean_data, expand, expected):
    "Tests the Deviation normalisation class for dask."

    mean_file_path = tmp_path / "mean.npy"
    stdev_file_path = tmp_path / "stdev.npy"
    _write_data(file_path=mean_file_path, data=mean_data)
    _write_data(file_path=stdev_file_path, data=0.1 * (mean_data + 1))  # +1 to avoid divide by 0

    a = Deviation(mean=mean_file_path, deviation=stdev_file_path, expand=expand)

    result = a.apply_func(sample)
    np.testing.assert_array_almost_equal(result.compute(), np.array(expected))

    result_reversed = a.undo_func(result)
    np.testing.assert_array_almost_equal(result_reversed.compute(), sample.compute())


@pytest.mark.parametrize(
    ("div_data", "expand", "expected"),
    (
        (np.array(range(1, 4)), True, [[0, 1], [1, 1.5], [4 / 3, 5 / 3]]),  # mean broadcasted over dim=0
        (np.array(range(1, 3)), False, [[0, 0.5], [2, 1.5], [4, 2.5]]),  # mean broadcasted over dim=-1
        (np.array(range(6, 12)).reshape(3, 2), True, [[0, 1 / 7], [2 / 8, 1 / 3], [2 / 5, 5 / 11]]),
    ),
)
def test_Division(sample, tmp_path, div_data, expand, expected):
    "Tests the Division normalisation class for dask."

    file_path = tmp_path / "div.npy"
    _write_data(file_path=file_path, data=div_data)

    a = Division(division_factor=file_path, expand=expand)

    result = a.apply_func(sample)
    np.testing.assert_array_almost_equal(result.compute(), np.array(expected))

    result_reversed = a.undo_func(result)
    np.testing.assert_array_almost_equal(result_reversed.compute(), sample.compute())


@pytest.mark.parametrize(
    ("norm_str", "denorm_str", "kwargs", "expected"),
    (
        ("sample + 1", "sample - 1", {}, np.array(range(1, 7)).reshape(3, 2)),
        ("sample + var", "sample - var", {"var": 5}, np.array(range(5, 11)).reshape(3, 2)),
        ("sample + var", "sample - var", {"var": "file"}, np.array([[0, 2], [2, 4], [4, 6]])),
    ),
)
def test_Evaluated(sample, tmp_path, norm_str, denorm_str, kwargs, expected):
    "Tests the Evaluated normalisation class for dask."

    if kwargs.get("var") == "file":
        file_path = tmp_path / "some_vals.npy"
        _write_data(file_path, range(2))
        kwargs["var"] = file_path

    e = Evaluated(normalisation_eval=norm_str, unnormalisation_eval=denorm_str, **kwargs)

    result = e.apply_func(sample)

    # Check if result is dask array and compute if so. E.g. sample + 1 might not be evaluated with dask depending on `eval`, but if it is, we need to compute.
    if isinstance(result, da.Array):
        result_comp = result.compute()
    else:
        result_comp = result

    np.testing.assert_array_equal(result_comp, expected)

    result_reverse = e.undo_func(result)

    if isinstance(result_reverse, da.Array):
        result_reverse_comp = result_reverse.compute()
    else:
        result_reverse_comp = result_reverse

    np.testing.assert_array_equal(result_reverse_comp, sample.compute())
