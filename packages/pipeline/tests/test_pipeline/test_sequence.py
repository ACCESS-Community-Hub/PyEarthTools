# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
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


from __future__ import annotations
from typing import Any

import pytest

import pyearthtools.utils

pyearthtools.utils.config.set({"pipeline.run_parallel": False})

import pyearthtools.data

from pyearthtools.pipeline import Pipeline, exceptions, modifications
from pyearthtools.pipeline.modifications.idx_modification import SequenceRetrieval

from tests.fake_pipeline_steps import *


@pytest.mark.parametrize(
    "samples, result",
    [
        (0, 0),
        (1, (0, 1)),
        (-1, (-1, 0)),
        (-2, (-2, 0)),
        (6, (0, 6)),
        (-6, (-6, 0)),
    ],
)
def test_sequence_int(samples, result):
    """Test integer behaviour"""
    pipe = Pipeline(FakeIndex(), SequenceRetrieval(samples))
    assert pipe[0] == result  # type: ignore


@pytest.mark.parametrize(
    "samples, result",
    [
        (0, 1),
        (1, 3),
        (-1, 1),
        (-2, 0),
        (6, 8),
        (-6, -4),
    ],
)
def test_sequence_int_merged(samples, result):
    """Test integer behaviour"""
    pipe = Pipeline(FakeIndex(), SequenceRetrieval(samples, merge_function=sum))
    assert pipe[1] == result  # type: ignore


@pytest.mark.parametrize(
    "samples, result",
    [
        ([-2, 1], -2),
        ([0, 3], (0, 1, 2)),
        ([-1, 2], (-1, 0)),
        ([-2, 2], (-2, -1)),
        ([-2, 3], (-2, -1, 0)),
        ([2, 3], (2, 3, 4)),
        # Different interval
        ([2, 3, 2], (2, 4, 6)),
        ([2, 3, 3], (2, 5, 8)),
        ([-10, 3, 3], (-10, -7, -4)),
    ],
)
def test_sequence_sequence(samples, result):
    """Test sequence"""
    pipe = Pipeline(FakeIndex(), SequenceRetrieval(samples))
    assert pipe[0] == result  # type: ignore


@pytest.mark.parametrize(
    "samples, result",
    [
        (
            [(-3, 2), 1],
            (
                (-3, -2),
                -1,
            ),
        ),
        ([(-3, 2), 2], ((-3, -2), 0)),
        (
            [(1, 2), 1],
            (
                (1, 2),
                3,
            ),
        ),
        (
            [(-3, 2), (2, 1)],
            ((-3, -2), 0),
        ),
        (
            [(-3, 2), (2, 2)],
            ((-3, -2), (0, 1)),
        ),
        (
            [(-3, 2), (2, 2), (1, 2)],
            (
                (-3, -2),
                (0, 1),
                (2, 3),
            ),
        ),
        (((0, 3), (1, 2)), ((0, 1, 2), (3, 4))),
        (((0, 3), (-1, 2)), ((0, 1, 2), (1, 2))),
        # Intervals
        (
            [(-3, 2, 2), (2, 2)],
            ((-3, -1), (1, 2)),
        ),
        (
            [(-3, 2, 3), (2, 2)],
            ((-3, 0), (2, 3)),
        ),
        (((0, 3, 2), (-1, 2)), ((0, 2, 4), (3, 4))),
    ],
)
def test_sequence_nested(samples, result):
    """Test nested"""
    pipe = Pipeline(FakeIndex(), SequenceRetrieval(samples))

    assert pipe[0] == result  # type: ignore
