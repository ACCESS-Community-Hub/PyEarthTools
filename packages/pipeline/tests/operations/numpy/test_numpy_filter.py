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

from pyearthtools.pipeline.operations.numpy import filters
from pyearthtools.pipeline.exceptions import PipelineFilterException

import numpy as np
import pytest


def test_DropAnyNan_false():

    original = np.array([[1, 2], [4, 3]])

    drop = filters.DropAnyNan()
    # No return value, just check no exception is raised
    drop.filter(original)


def test_DropAnyNan_true():

    original = np.array([[1, 2], [4, np.nan]])

    drop = filters.DropAnyNan()

    with pytest.raises(PipelineFilterException):
        drop.filter(original)


def test_DropAllNan_false():

    original = np.array([[1, 2], [np.nan, 3]])

    drop = filters.DropAllNan()
    # No return value, just check no exception is raised
    drop.filter(original)


def test_DropAllNan_true():

    original = np.array([[np.nan, np.nan], [np.nan, np.nan]])

    drop = filters.DropAllNan()

    with pytest.raises(PipelineFilterException):
        drop.filter(original)


def test_DropValue():

    # test drop case
    original = np.array([[1, 1], [np.nan, np.nan]])

    drop = filters.DropValue(value=1, percentage=75)

    with pytest.raises(PipelineFilterException):
        drop.filter(original)

    # test no drop case
    drop = filters.DropValue(value=1, percentage=50)
    drop.filter(original)

    # test with nan - drop case
    drop = filters.DropValue(value="nan", percentage=75)

    with pytest.raises(PipelineFilterException):
        drop.filter(original)

    # no drop case
    drop = filters.DropValue(value="nan", percentage=50)
    drop.filter(original)
