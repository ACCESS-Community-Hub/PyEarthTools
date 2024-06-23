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

from edit.pipeline_V2 import config

config.RUN_PARALLEL = False

from edit.pipeline_V2 import Pipeline, exceptions, branching, exceptions

from tests.fake_pipeline_steps import *


def test_branchingpoint_basic():
    pipe = Pipeline((FakeIndex(), FakeIndex()))
    assert pipe[1] == (1, 1)


def test_branch_differing_operations():
    pipe = Pipeline(FakeIndex(), (MultiplicationOperation(10), MultiplicationOperation(2)))
    assert pipe[1] == (10, 2)


def test_branch_differing_operations_larger():
    pipe = Pipeline(
        FakeIndex(),
        (
            (MultiplicationOperation(10), MultiplicationOperation(5)),
            MultiplicationOperation(2),
        ),
    )
    assert pipe[1] == (50, 2)


def test_branch_differing_operations_larger_direct():
    pipe = Pipeline(
        FakeIndex(),
        (
            (MultiplicationOperation(10), MultiplicationOperation(5)),
            MultiplicationOperation(2),
        ),
    )
    assert pipe[1] == (50, 2)


def test_branch_differing_operations_nested():
    pipe = Pipeline(
        FakeIndex(),
        (
            ((MultiplicationOperation(10), MultiplicationOperation(5)),),
            MultiplicationOperation(2),
        ),
    )
    assert pipe[1] == ((10, 5), 2)


def test_branch_differing_operations_nested_larger():
    pipe = Pipeline(
        FakeIndex(),
        (
            (
                (
                    (MultiplicationOperation(10), MultiplicationOperation(10)),
                    MultiplicationOperation(5),
                ),
            ),
            MultiplicationOperation(2),
        ),
    )
    assert pipe[1] == ((100, 5), 2)


def test_branch_differing_operations_undo():
    pipe = Pipeline(FakeIndex(), (MultiplicationOperation(10), MultiplicationOperation(2)))
    assert pipe.undo(pipe[1]) == (1, 1)


def test_branch_differing_operations_undo_unify():
    pipe = Pipeline(
        FakeIndex(),
        branching.unify.Equality(),
        (MultiplicationOperation(10), MultiplicationOperation(2)),
    )
    assert pipe.undo(pipe[1]) == 1


def test_branch_differing_operations_undo_unify_fail():
    pipe = Pipeline(
        FakeIndex(),
        branching.unify.Equality(),
        (MultiplicationOperationUnunifiedable(10), MultiplicationOperation(2)),
    )
    with pytest.raises(exceptions.PipelineUnificationException):
        assert pipe.undo(pipe[1]) == 1


def test_branch_differing_sources():
    pipe = Pipeline(
        (FakeIndex(2), FakeIndex()),
        MultiplicationOperation(10),
    )
    assert pipe[1] == (20, 10)


def test_branch_differing_sources_undo():
    pipe = Pipeline(
        (FakeIndex(2), FakeIndex()),
        MultiplicationOperation(10),
    )
    assert pipe.undo(pipe[1]) == (2, 1)


def test_branch_differing_sources_with_steps():
    pipe = Pipeline(
        (
            (FakeIndex(2), MultiplicationOperation(2)),
            (FakeIndex(), MultiplicationOperation(3)),
        ),
        MultiplicationOperation(10),
    )
    assert pipe[1] == (40, 30)


def test_branch_differing_sources_with_steps_undo():
    pipe = Pipeline(
        (
            (FakeIndex(2), MultiplicationOperation(2)),
            (FakeIndex(), MultiplicationOperation(3)),
        ),
        MultiplicationOperation(10),
    )
    assert pipe.undo(pipe[1]) == (2, 1)


def test_branch_with_invalid():
    with pytest.raises(exceptions.PipelineTypeError):
        pipe = Pipeline(
            ((FakeIndex(2), MultiplicationOperation(2)), (FakeIndex(), lambda x: x)),
            MultiplicationOperation(10),
        )


def test_branch_with_mapping():
    pipe = Pipeline(
        (FakeIndex(), FakeIndex()),
        (MultiplicationOperation(1), MultiplicationOperation(2), "map"),
    )
    assert pipe[1] == (1, 2)


def test_branch_with_mapping_not_tuple():
    pipe = Pipeline(
        FakeIndex(),
        (MultiplicationOperation(1), MultiplicationOperation(2), "map"),
    )
    with pytest.raises(exceptions.PipelineRuntimeError):
        assert pipe[1] == (1, 2)


def test_branch_with_mapping_wrong_size():
    pipe = Pipeline(
        (FakeIndex(), FakeIndex()),
        (
            MultiplicationOperation(1),
            MultiplicationOperation(2),
            MultiplicationOperation(3),
            "map",
        ),
    )
    with pytest.raises(exceptions.PipelineRuntimeError):
        assert pipe[1] == (1, 2)
