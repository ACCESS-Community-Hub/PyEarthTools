# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations

import pytest

import edit.utils

from edit.pipeline_V2 import Pipeline, modifications

from tests.fake_pipeline_steps import FakeIndex, MultiplicationOperation  # noqa: F403

edit.utils.config.set({'pipeline_V2.run_parallel': False})


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
        FakeIndex("test"), # type: ignore
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
    import edit.data

    pipe = Pipeline(FakeIndex(), modifications.TimeIdxModifier("6 hours"))
    assert pipe[edit.data.EDITDatetime("2000-01-01T00")] == edit.data.EDITDatetime("2000-01-01T06")


# def test_TimeIdxModifier_basic_tuple():
#     import edit.data
#     pipe = Pipeline(FakeIndex(), pipelines.TimeIdxModifier((6, 'hours')))
#     assert pipe[edit.data.EDITDatetime('2000-01-01T00')] == edit.data.EDITDatetime('2000-01-01T06')


def test_TimeIdxModifier_nested():
    import edit.data

    pipe = Pipeline(FakeIndex(), modifications.TimeIdxModifier(("6 hours", "12 hours")))
    assert pipe[edit.data.EDITDatetime("2000-01-01T00")] == (
        edit.data.EDITDatetime("2000-01-01T06"),
        edit.data.EDITDatetime("2000-01-01T12"),
    )
