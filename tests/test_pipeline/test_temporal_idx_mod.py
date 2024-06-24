# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations
from typing import Any

import pytest

import edit.utils
edit.utils.config.set({'pipeline_V2.run_parallel': False})

import edit.data

from edit.pipeline_V2 import Pipeline, exceptions
from edit.pipeline_V2.modifications.idx_modification import TemporalRetrieval

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
