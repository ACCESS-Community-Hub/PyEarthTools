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

from pyearthtools.pipeline import Pipeline, exceptions, branching

from tests.fake_pipeline_steps import FakeIndex, MultiplicationOperation, MultiplicationOperationUnunifiedable

pyearthtools.utils.config.set({"pipeline.run_parallel": False})


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
    assert pipe.undo(pipe[1]) == 1


# def test_branch_differing_operations_undo_unify():
#     pipe = Pipeline(
#         FakeIndex(),
#         branching.unify.Equality(),
#         (MultiplicationOperation(10), MultiplicationOperation(2)),
#     )
#     assert pipe.undo(pipe[1]) == 1


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
        _ = Pipeline(
            ((FakeIndex(2), MultiplicationOperation(2)), (FakeIndex(), lambda x: x)),
            MultiplicationOperation(10),
        )


def test_branch_with_mapping():
    pipe = Pipeline(
        (FakeIndex(), FakeIndex()),
        (MultiplicationOperation(1), MultiplicationOperation(2), "map"),
    )
    assert pipe[1] == (1, 2)


def test_branch_with_mapping_copy():
    pipe = Pipeline(
        (FakeIndex(), FakeIndex(2)),
        (MultiplicationOperation(1), "map_copy"),
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


def test_branch_with_source():
    pipe = Pipeline(
        FakeIndex(),
        (MultiplicationOperation(2), FakeIndex()),
    )
    assert pipe[1] == (2, 1)
