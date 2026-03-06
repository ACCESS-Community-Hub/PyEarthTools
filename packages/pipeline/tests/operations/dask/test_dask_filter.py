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

from pyearthtools.pipeline.operations.dask import filters
from pyearthtools.pipeline.exceptions import PipelineFilterException

import numpy as np
import dask.array as da
import pytest


def test_DropAnyNan():
    """Tests DropAnyNan dask filter."""

    original = da.ones((2, 2))

    # no nans - should succeed quietly
    drop = filters.DropAnyNan()
    drop.filter(original)

    # one nan - should raise exception
    original[0, 0] = np.nan
    drop = filters.DropAnyNan()
    with pytest.raises(PipelineFilterException):
        drop.filter(original)


# xfailed since the result seems to be inverted to documented requirements
@pytest.mark.xfail
def test_DropAllNan():
    """Tests DropAllNan dask filter."""

    original = da.empty((2, 2))

    # no nans - should succeed quietly
    drop = filters.DropAllNan()
    drop.filter(original)

    # one nan - should succeed quietly
    original[0, 0] = np.nan
    drop.filter(original)

    # all nans - should raise exception
    original[:, :] = np.nan
    with pytest.raises(PipelineFilterException):
        drop.filter(original)


def test_DropValue():
    """Tests DropValue dask filter."""

    original = da.from_array([[0, 0], [1, 2]])

    # non-drop case (num zeros < threshold)
    drop = filters.DropValue(0, 75)
    drop.filter(original)

    # drop case  (num zeros >= threshold)
    drop = filters.DropValue(0, 50)
    with pytest.raises(PipelineFilterException):
        drop.filter(original)

    # non-drop case  (num nans < threshold)
    original = da.from_array([[np.nan, np.nan], [1, 2]])
    drop = filters.DropValue("nan", 75)
    drop.filter(original)

    # drop case (num nans >= threshold)
    drop = filters.DropValue("nan", 50)
    with pytest.raises(PipelineFilterException):
        drop.filter(original)


def test_Shape():
    """Tests Shape dask filter."""

    originals = (da.empty((2, 2)), da.empty((2, 3)))

    # check drop case
    drop = filters.Shape((2, 3))
    with pytest.raises(PipelineFilterException):
        drop.filter(originals[0])

    # check non-drop case
    drop = filters.Shape((2, 2))
    drop.filter(originals[0])

    # check tuple inputs drop cases
    drop = filters.Shape(((2, 3), (2, 3)))
    with pytest.raises(PipelineFilterException):
        drop.filter(originals)

    # check tuple inputs non-drop cases
    drop = filters.Shape(((2, 2), (2, 3)))
    drop.filter(originals)

    # invalid mismatched shape and input
    drop = filters.Shape(((2, 2),))
    with pytest.raises(RuntimeError):
        drop.filter(originals)
