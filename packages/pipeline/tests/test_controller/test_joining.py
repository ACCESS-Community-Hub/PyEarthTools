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

from pyearthtools.pipeline import Pipeline, branching

from tests.fake_pipeline_steps import FakeIndex, MultiplicationOperation

pyearthtools.utils.config.set({"pipeline.run_parallel": False})


class AdditionJoin(branching.Joiner):
    def join(self, sample: tuple) -> Any:
        return sum(sample)

    def unjoin(self, sample: Any) -> tuple:
        return super().unjoin(sample)


class AdditionUnJoin(branching.Joiner):
    def join(self, sample: tuple) -> Any:
        self._record = sample[0]
        return sum(sample)

    def unjoin(self, sample: Any) -> tuple:
        return (self._record, sample - self._record)


def test_branch_with_join_invalid():
    pipe = Pipeline(
        FakeIndex(2),
        MultiplicationOperation(10),
        AdditionJoin(),
    )
    with pytest.raises(TypeError):
        assert pipe[1] == 20


def test_branch_with_join():
    pipe = Pipeline(
        (FakeIndex(2), FakeIndex()),
        MultiplicationOperation(10),
        AdditionJoin(),
    )
    assert pipe[1] == 30


def test_branch_with_join_undo_pass():
    pipe = Pipeline(
        (FakeIndex(2), FakeIndex()),
        MultiplicationOperation(10),
        AdditionJoin(),
    )
    assert pipe.undo(pipe[1]) == 3


def test_branch_with_join_undo():
    pipe = Pipeline(
        (FakeIndex(2), FakeIndex()),
        MultiplicationOperation(10),
        AdditionUnJoin(),
    )
    assert pipe.undo(pipe[1]) == (2, 1)


class AdditionJoinUnImplemented(branching.Joiner): ...


@pytest.mark.parametrize(
    "operation",
    [
        ("apply"),
        ("undo"),
        ("both"),
    ],
)
def test_branch_with_join_unimplemented(operation):
    with pytest.raises(TypeError):
        _ = Pipeline(
            (FakeIndex(2), FakeIndex()),
            MultiplicationOperation(10),
            AdditionJoinUnImplemented(operation=operation),  # type: ignore
        )
