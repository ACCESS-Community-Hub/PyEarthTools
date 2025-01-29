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

from pyearthtools.pipeline import Pipeline, exceptions
from pyearthtools.pipeline.modifications.idx_modification import TemporalRetrieval

from tests.fake_pipeline_steps import *


def map_to_str(t):
    if isinstance(t, tuple):
        return tuple(map(map_to_str, t))
    return str(t)


@pytest.mark.parametrize(
    "samples, time, result",
    [
        (0, "2020-01-01", "2020-01-01"),
        (1, "2020-01-01", ("2020-01-01", "2020-01-02")),
        (-1, "2020-01-02", ("2020-01-01", "2020-01-02")),
        (-2, "2020-01-16", ("2020-01-14", "2020-01-16")),
        (6, "2020-01-01T00", ("2020-01-01T00", "2020-01-01T06")),
    ],
)
def test_Temporal_int(samples, time, result):
    """Test integer behaviour"""
    pipe = Pipeline(FakeIndex(), TemporalRetrieval(samples))
    assert map_to_str(pipe[time]) == result  # type: ignore


@pytest.mark.parametrize(
    "samples, time, result",
    [
        ([-2, 1], "2020-01-16", "2020-01-14"),
        ([-2, 2], "2020-01-16", ("2020-01-14", "2020-01-15")),
        ([-2, 3], "2020-01-16", ("2020-01-14", "2020-01-15", "2020-01-16")),
        ([2, 3], "2020-01-16", ("2020-01-18", "2020-01-19", "2020-01-20")),
    ],
)
def test_Temporal_sequence(samples, time, result):
    """Test sequence"""
    pipe = Pipeline(FakeIndex(), TemporalRetrieval(samples))
    assert map_to_str(pipe[time]) == result  # type: ignore


@pytest.mark.parametrize(
    "samples, time, result",
    [
        ([(-3, 2), 1], "2020-01-16", (("2020-01-13", "2020-01-14"), "2020-01-15")),
        ([(1, 2), 1], "2020-01-16", (("2020-01-17", "2020-01-18"), "2020-01-19")),
        (
            [(-3, 2), (2, 1)],
            "2020-01-16",
            (("2020-01-13", "2020-01-14"), "2020-01-16"),
        ),
        (
            [(-3, 2), (2, 2)],
            "2020-01-16",
            (("2020-01-13", "2020-01-14"), ("2020-01-16", "2020-01-17")),
        ),
        (
            [(-3, 2), (2, 2), (1, 2)],
            "2020-01-16",
            (
                ("2020-01-13", "2020-01-14"),
                ("2020-01-16", "2020-01-17"),
                ("2020-01-18", "2020-01-19"),
            ),
        ),
    ],
)
def test_Temporal_nested(samples, time, result):
    """Test nested"""
    pipe = Pipeline(FakeIndex(), TemporalRetrieval(samples))

    assert map_to_str(pipe[time]) == result  # type: ignore
