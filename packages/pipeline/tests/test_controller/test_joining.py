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
