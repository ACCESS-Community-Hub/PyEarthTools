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


def test_DropValue():
    """Tests DropValue xarray filter."""

    original = xr.Dataset(
        {"var1": xr.DataArray(np.array([[1, 1], [3, 4]])), "var2": xr.DataArray(np.array([[np.nan, np.nan], [6, 7]]))}
    )

    # check var1 of dataset drop case
    drop = filters.DropValue(1, 75)
    with pytest.raises(PipelineFilterException):
        drop.filter(original["var1"])

    # check var1 of dataset non-drop case
    drop = filters.DropValue(1, 50)
    drop.filter(original["var1"])

    # check var2 of dataset drop case (using nan)
    drop = filters.DropValue("nan", 75)
    with pytest.raises(PipelineFilterException):
        drop.filter(original["var2"])

    # check var2 of dataset non-drop case
    drop = filters.DropValue("nan", 50)
    drop.filter(original["var2"])

    # check whole dataset drop case
    drop = filters.DropValue(1, 50)
    with pytest.raises(PipelineFilterException):
        drop.filter(original)

    # check whole dataset non-drop case
    drop = filters.DropValue(1, 10)
    drop.filter(original)

    # check whole dataset nan drop case
    drop = filters.DropValue("nan", 50)
    with pytest.raises(PipelineFilterException):
        drop.filter(original)

    # check whole dataset nan non-drop case
    drop = filters.DropValue("nan", 10)
    drop.filter(original)

    # check invalid type
    with pytest.raises(TypeError):
        drop.filter(np.empty((1, 1)))


def test_Shape():
    """Tests Shape xarray filter."""

    originals = (
        xr.Dataset({"var": xr.DataArray(np.empty((2, 2)))}),
        xr.Dataset({"var": xr.DataArray(np.empty((2, 3)))}),
    )

    # check DataArray drop case
    drop = filters.Shape((2, 3))
    with pytest.raises(PipelineFilterException):
        drop.filter(originals[0]["var"])

    # check Dataset drop case
    with pytest.raises(PipelineFilterException):
        drop.filter(originals[0])

    # check non-drop cases
    drop = filters.Shape((2, 2))
    drop.filter(originals[0]["var"])
    drop = filters.Shape((1, 2, 2))
    drop.filter(originals[0])

    # check tuple inputs drop cases
    drop = filters.Shape(((1, 2, 3), (1, 2, 3)))
    with pytest.raises(PipelineFilterException):
        drop.filter(originals)

    drop = filters.Shape(((2, 3), (2, 2)))
    with pytest.raises(PipelineFilterException):
        drop.filter(tuple(ds["var"] for ds in originals))

    # check tuple inputs non-drop cases
    drop = filters.Shape(((1, 2, 2), (1, 2, 3)))
    drop.filter(originals)

    drop = filters.Shape(((2, 2), (2, 3)))
    drop.filter(tuple(ds["var"] for ds in originals))

    # invalid mismatched shape and input
    drop = filters.Shape(((1, 2, 2),))
    with pytest.raises(RuntimeError):
        drop.filter(originals)

    # try invalid input type
    drop = filters.Shape((2,))
    with pytest.raises(TypeError):
        drop.filter([1, 2])
