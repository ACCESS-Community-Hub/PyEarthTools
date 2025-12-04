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

from pyearthtools.pipeline.operations.xarray import filters
from pyearthtools.pipeline.exceptions import PipelineFilterException

import numpy as np
import xarray as xr
import pytest


def test_DropAnyNan():
    """Tests DropAnyNan xarray filter."""

    original = xr.Dataset(
        {"var1": xr.DataArray(np.array([[1, 2], [3, 4]])), "var2": xr.DataArray(np.array([[np.nan, 5], [6, 7]]))}
    )

    # check var1 of dataset - should succeed quietly
    drop = filters.DropAnyNan("var1")
    drop.filter(original)

    # warning if dataarray is passed in with filter intiialized with variable
    with pytest.warns():
        drop.filter(original["var1"])

    # check var2 - should raise exception
    drop = filters.DropAnyNan("var2")
    with pytest.raises(PipelineFilterException):
        drop.filter(original["var2"])

    # check whole dataset - should raise excpetion
    drop = filters.DropAnyNan()
    with pytest.raises(PipelineFilterException):
        drop.filter(original)

    # check whole dataset without nans - should succeed quietly
    original["var2"][0, 0] = 0
    drop.filter(original)

    # test wrong type
    with pytest.raises(TypeError):
        drop.filter(np.empty(1))


def test_DropAllNan():
    """Tests DropAllNan xarray filter."""

    original = xr.Dataset(
        {
            "var1": xr.DataArray(np.array([[np.nan, np.nan], [np.nan, 4]])),
            "var2": xr.DataArray(np.array([[np.nan, np.nan], [np.nan, np.nan]])),
        }
    )

    # check var1 of dataset - should succeed quietly
    drop = filters.DropAllNan("var1")
    drop.filter(original)

    # warning if dataarray is passed in with filter initialized with variable
    with pytest.warns():
        drop.filter(original["var1"])

    # check var2 - should raise exception
    drop = filters.DropAllNan("var2")
    with pytest.raises(PipelineFilterException):
        drop.filter(original["var2"])

    # check whole dataset - should succeed quietly
    drop = filters.DropAllNan()
    drop.filter(original)

    # check whole dataset without nans - should succeed quietly
    original["var2"][0, 0] = 0
    drop.filter(original)

    # test wrong type
    with pytest.raises(TypeError):
        drop.filter(np.empty(1))
