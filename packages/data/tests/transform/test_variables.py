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

import pytest
import xarray as xr

from pyearthtools.data.transforms.variables import Drop, Select, Trim


def test_trim():
    data = xr.Dataset(
        {
            "var_to_keep": ("time", [0, 0, 7]),
            "var_to_ignore1": ("time", [0, 0, 7]),
            "var_to_ignore2": ("time", [0, 0, 7]),
        }
    )

    # Test variable drops var as intended
    orig_data = data.copy()
    transform = Trim("var_to_keep")

    transformed_data = transform.apply(data)
    assert "var_to_ignore1" not in transformed_data.data_vars
    assert "var_to_ignore2" not in transformed_data.data_vars

    # Check transformed dataset hasn't been modified
    assert data.equals(orig_data)

    # Test a var that is not in the list returns the same dataset
    transform = Trim("var_not_found")
    transformed_data = transform.apply(data)
    assert transformed_data.equals(orig_data)

    # Test None returns the same dataset
    transform = Trim(None)
    transformed_data = transform.apply(data)
    assert transformed_data.equals(orig_data)


def test_drop():
    data = xr.Dataset(
        {
            "var_to_keep": ("time", [0, 0, 7]),
            "var_to_drop": ("time", [0, 0, 7]),
        }
    )

    # Test variable drops var as intended
    orig_data = data.copy()
    transform = Drop("var_to_drop")
    transformed_data = transform.apply(data)
    assert "var_to_drop" not in transformed_data.data_vars
    # We don't want to modify the original data, so testing it's not in place
    assert data.equals(orig_data)

    # Test empty list returns the same dataset
    transform = Drop([])
    transformed_data = transform.apply(data)
    assert data.equals(transformed_data)

    # Test a var that is not in the list
    with pytest.raises(ValueError):
        transform = Drop("var_not_found")
        _ = transform.apply(data)

    # Test None raise valueError
    with pytest.raises(ValueError):
        transform = Drop(None)
        _ = transform.apply(data)


def test_select():
    data = xr.Dataset(
        {
            "var_to_keep": ("time", [0, 0, 7]),
            "var_to_ignore1": ("time", [0, 0, 7]),
            "var_to_ignore2": ("time", [0, 0, 7]),
        }
    )
    orig_data = data.copy()

    transform = Select("var_to_keep")
    transformed_data = transform.apply(data)
    assert "var_to_ignore1" not in transformed_data.data_vars
    assert "var_to_ignore2" not in transformed_data.data_vars

    # Test empty list returns the same dataset
    transform = Select([])
    transformed_data = transform.apply(data)
    assert orig_data.equals(transformed_data)

    # Test a var that is not in the list returns the same dataset
    transform = Select("var_not_found")
    transformed_data = transform.apply(data)
    assert transformed_data.equals(orig_data)

    # Test None returns the same dataset
    transform = Select(None)
    transformed_data = transform.apply(data)
    assert transformed_data.equals(orig_data)
