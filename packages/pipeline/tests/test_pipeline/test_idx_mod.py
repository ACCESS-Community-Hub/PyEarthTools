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

from pyearthtools.pipeline import Pipeline, modifications

from tests.fake_pipeline_steps import FakeIndex, MultiplicationOperation  # noqa: F403

pyearthtools.utils.config.set({"pipeline.run_parallel": False})


def test_IdxModifier_basic():
    pipe = Pipeline(FakeIndex(), modifications.IdxModifier((0,)))
    assert pipe[0] == (0,)


def test_IdxModifier_basic_no_tuple():
    pipe = Pipeline(FakeIndex(), modifications.IdxModifier(0))
    assert pipe[0] == 0


def test_IdxModifier_two_samples():
    pipe = Pipeline(FakeIndex(), modifications.IdxModifier((0, 1)))
    assert pipe[0] == (0, 1)


def test_IdxModifier_nested():
    pipe = Pipeline(FakeIndex(), modifications.IdxModifier((0, (1, 2))))
    assert pipe[0] == (0, (1, 2))


def test_IdxModifier_nested_double():
    pipe = Pipeline(FakeIndex(), modifications.IdxModifier((0, (1, (2, 3)))))
    assert pipe[0] == (0, (1, (2, 3)))


def test_IdxModifier_nested_merge():
    pipe = Pipeline(FakeIndex(), modifications.IdxModifier((0, (1, 2)), merge=True, merge_function=sum))
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
        modifications.IdxModifier((1, (2, (3, 4))), merge=depth, merge_function=sum),
    )
    assert pipe[0] == result


def test_IdxModifier_unmergeable():
    pipe = Pipeline(
        FakeIndex("test"),  # type: ignore
        modifications.IdxModifier(("t", "a"), merge=True, merge_function=sum),
    )
    with pytest.raises(TypeError):
        assert pipe[1] == (1, 5)


def test_IdxMod_stacked():
    pipe = Pipeline(
        FakeIndex(),
        modifications.IdxModifier((0, 1)),
        modifications.IdxModifier((0, 1)),
    )
    assert pipe[1] == ((1, 2), (2, 3))


def test_IdxMod_stacked_with_mult():
    pipe = Pipeline(
        FakeIndex(),
        modifications.IdxModifier((0, 1)),
        modifications.IdxModifier((0, 1)),
        MultiplicationOperation(2),
    )
    assert pipe[1] == ((2, 4), (4, 6))


def test_IdxMod_with_branch():
    pipe = Pipeline(
        FakeIndex(),
        modifications.IdxModifier((0, 1)),
        (
            (MultiplicationOperation(1),),
            (MultiplicationOperation(2),),
        ),
    )
    assert pipe[1] == ((1, 2), (2, 4))


def test_IdxMod_with_branch_mapping():
    pipe = Pipeline(
        FakeIndex(),
        modifications.IdxModifier((0, 1)),
        ((MultiplicationOperation(1),), (MultiplicationOperation(2),), "map"),
    )
    assert pipe[1] == (1, 4)


#### Idx Override


def test_IdxOverride_basic():
    pipe = Pipeline(FakeIndex(), modifications.IdxOverride(0))
    assert pipe[1] == 0


#### TimeIdxModifier


def test_TimeIdxModifier_basic():
    import pyearthtools.data

    pipe = Pipeline(FakeIndex(), modifications.TimeIdxModifier("6 hours"))
    assert pipe[pyearthtools.data.pyearthtoolsDatetime("2000-01-01T00")] == pyearthtools.data.pyearthtoolsDatetime("2000-01-01T06")


# def test_TimeIdxModifier_basic_tuple():
#     import pyearthtools.data
#     pipe = Pipeline(FakeIndex(), pipelines.TimeIdxModifier((6, 'hours')))
#     assert pipe[pyearthtools.data.pyearthtoolsDatetime('2000-01-01T00')] == pyearthtools.data.pyearthtoolsDatetime('2000-01-01T06')


def test_TimeIdxModifier_nested():
    import pyearthtools.data

    pipe = Pipeline(FakeIndex(), modifications.TimeIdxModifier(("6 hours", "12 hours")))
    assert pipe[pyearthtools.data.pyearthtoolsDatetime("2000-01-01T00")] == (
        pyearthtools.data.pyearthtoolsDatetime("2000-01-01T06"),
        pyearthtools.data.pyearthtoolsDatetime("2000-01-01T12"),
    )
