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

import pytest

import pyearthtools.utils

from pyearthtools.pipeline import Pipeline, Operation
from pyearthtools.data import Index, Petdt
from pyearthtools.pipeline.modifications import (
    IdxModifier,
    IdxOverride,
    TimeIdxModifier,
    TemporalWindow,
    TemporalRetrieval,
)
from pyearthtools.data.time import TimeDelta
import xarray as xr
import numpy as np


class FakeIndex(Index):
    """Simply returns the `idx` or `override`."""

    def __init__(self, override: int | None = None):
        self._overrideValue = override
        super().__init__()

    def get(self, idx):
        return self._overrideValue or idx


class MultiplicationOperation(Operation):
    def __init__(self, factor):
        super().__init__(split_tuples=True)
        self.factor = factor

    def apply_func(self, sample):
        return sample * self.factor

    def undo_func(self, sample):
        return sample // self.factor


pyearthtools.utils.config.set({"pipeline.run_parallel": False})


def test_multiplication_undo():
    mo = MultiplicationOperation(2)
    mo2 = mo.apply_func(2)
    orig = mo.undo_func(mo2)
    assert orig == 2


@pytest.mark.parametrize("mod", ((0,), 0, (0, 1), (0, (1, 2)), (0, (1, (2, 3)))))
def test_IdxModifier_basic(mod):
    pipe = Pipeline(FakeIndex(), IdxModifier(mod))
    assert pipe[0] == mod
    # check that extra_mods gets passed through
    if type(mod) is tuple and len(mod) > 1:
        pipe = Pipeline(FakeIndex(), IdxModifier(*mod))
        assert pipe[0] == mod


def test_IdxModifier_nested_merge():
    pipe = Pipeline(FakeIndex(), IdxModifier((0, (1, 2)), merge=True, merge_function=sum))
    assert pipe[0] == (0, 3)


@pytest.mark.parametrize(
    "depth, result",
    [
        (0, (1, (2, (3, 4)))),
        (1, (1, (2, 7))),
        (2, (1, 9)),
        (3, 10),
    ],
)
def test_IdxModifier_merge_depth(depth, result):
    pipe = Pipeline(
        FakeIndex(),
        IdxModifier((1, (2, (3, 4))), merge=depth, merge_function=sum),
    )
    assert pipe[0] == result


def test_IdxModifier_unmergeable():
    pipe = Pipeline(
        FakeIndex("test"),  # type: ignore
        IdxModifier(("t", "a"), merge=True, merge_function=sum),
    )
    with pytest.raises(TypeError):
        assert pipe[1] == (1, 5)


def test_IdxMod_stacked():
    pipe = Pipeline(
        FakeIndex(),
        IdxModifier((0, 1)),
        IdxModifier((0, 1)),
    )
    assert pipe[1] == ((1, 2), (2, 3))


def test_IdxMod_stacked_with_mult():
    pipe = Pipeline(
        FakeIndex(),
        IdxModifier((0, 1)),
        IdxModifier((0, 1)),
        MultiplicationOperation(2),
    )
    assert pipe[1] == ((2, 4), (4, 6))


def test_IdxMod_with_branch():
    pipe = Pipeline(
        FakeIndex(),
        IdxModifier((0, 1)),
        (
            (MultiplicationOperation(1),),
            (MultiplicationOperation(2),),
        ),
    )
    assert pipe[1] == ((1, 2), (2, 4))


def test_IdxMod_with_branch_mapping():
    pipe = Pipeline(
        FakeIndex(),
        IdxModifier((0, 1)),
        ((MultiplicationOperation(1),), (MultiplicationOperation(2),), "map"),
    )
    assert pipe[1] == (1, 4)


#### Idx Override


def test_IdxOverride_basic():
    pipe = Pipeline(FakeIndex(), IdxOverride(0))
    assert pipe[1] == 0


#### TimeIdxModifier


def test_TimeIdxModifier_basic():

    pipe = Pipeline(FakeIndex(), TimeIdxModifier("6 hours"))
    assert pipe[Petdt("2000-01-01T00")] == Petdt("2000-01-01T06")


# def test_TimeIdxModifier_basic_tuple():
#     import pyearthtools.data
#     pipe = Pipeline(FakeIndex(), pipelines.TimeIdxModifier((6, 'hours')))
#     assert pipe[Petdt('2000-01-01T00')] == Petdt('2000-01-01T06')


def test_TimeIdxModifier_nested():

    pipe = Pipeline(FakeIndex(), TimeIdxModifier(("6 hours", "12 hours")))
    assert pipe[Petdt("2000-01-01T00")] == (
        Petdt("2000-01-01T06"),
        Petdt("2000-01-01T12"),
    )


def test_TimeIdxModifier_extramods():
    """Tests TimeIdxModifier with modifications passed as variable args (extra_mods)"""
    # first arg to TimeIdxModifier goes to "modifications" and second goes to extra_args
    pipe = Pipeline(FakeIndex(), TimeIdxModifier("6 hours", "12 hours"))
    assert pipe[Petdt("2000-01-01T00")] == (
        Petdt("2000-01-01T06"),
        Petdt("2000-01-01T12"),
    )


class test_data_accessor(Index):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data = {
            "2025-01-01": 1,
            "2025-01-02": 2,
            "2025-01-03": 3,
        }

    def get(self, time):
        return self._data[time]


@pytest.mark.parametrize("merge_method, expected", ((None, (1, 3)), (sum, 4)))
def test_TemporalRetrieval(merge_method, expected):
    """Test temporal retrieval."""

    # instantiate temporal retrieval with merge_method - retrieving from two steps behind.
    temporal_retrieval_step = TemporalRetrieval(-2, merge_function=merge_method, delta_unit="day")

    pipeline = Pipeline(test_data_accessor(), temporal_retrieval_step)

    assert expected == pipeline[Petdt("2025-01-03")]


def test_TemporalRetrieval_xarrayaccessor():
    """Tests temporal retrieval default merger for xarray data."""

    # insantiate TemporalRetrieval without merge method
    temporal_retrieval_step = TemporalRetrieval(-2)

    # create a data accessor with fake xarray data.
    data_accessor = test_data_accessor()
    data_accessor._data = {date: xr.DataArray([val] * 2, name=f"arr{val}") for date, val in data_accessor._data.items()}
    pipeline = Pipeline(data_accessor, temporal_retrieval_step)

    assert xr.merge((data_accessor["2025-01-01"], data_accessor["2025-01-03"])) == pipeline["2025-01-03"]


def test_TemporalRetrieval_npconcat():
    """Tests temporal retrieval default merger for numpy data."""

    # insantiate TemporalRetrieval with concat
    temporal_retrieval_step = TemporalRetrieval(-2, concat=True)

    # create a data accessor with fake numpy data.
    data_accessor = test_data_accessor()
    data_accessor._data = {date: np.array((val, val + 1)) for date, val in data_accessor._data.items()}
    pipeline = Pipeline(data_accessor, temporal_retrieval_step)

    assert np.array_equal(np.array((1, 2, 3, 4)), pipeline["2025-01-03"])


def test_TemporalRetrieval_invalid():
    """Tests errors when using/instantiating TemporalRetrieval."""
    with pytest.raises(ValueError):
        TemporalRetrieval(None)  # index ought to be int or iterable of ints
    tr = TemporalRetrieval(-1)
    with pytest.raises(TypeError):
        tr["a"]  # not convertable to Petdt

    # this is actually covering a type error in IdxModifier._run_merge
    invalid_accessor = test_data_accessor()
    invalid_accessor._data["2025-01-01"] = "a"
    pipeline = Pipeline(invalid_accessor, TemporalRetrieval(-2))
    with pytest.raises(TypeError):
        pipeline["2025-01-03"]


@pytest.mark.parametrize("merge_method, expected", ((None, ([1, 2], [3])), (sum, (3, 3))))
def test_TemporalWindow(merge_method, expected):
    """Test temporal window."""

    # instantiate temporal window with merge method
    temporal_window_step = TemporalWindow(
        prior_indexes=[-2, -1],
        posterior_indexes=[0],
        timedelta=TimeDelta((1, "day")),
        merge_method=merge_method,
    )

    # Instantiate pipeline with test data and temporal window
    pipeline = Pipeline(test_data_accessor(), temporal_window_step)
    assert expected == pipeline["2025-01-03"]
