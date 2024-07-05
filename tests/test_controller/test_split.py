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

from edit.pipeline import Pipeline, branching

from tests.fake_pipeline_steps import FakeIndex

edit.utils.config.set({"pipeline.run_parallel": False})


class Split(branching.Spliter):
    def split(self, sample):
        return (sample, sample)

    def join(self, sample):
        return super().join(sample)


def test_branch_with_split():
    pipe = Pipeline(
        FakeIndex(),
        Split(),
    )
    assert pipe[1] == (1, 1)


class SpliterUnImplemented(branching.Spliter): ...


@pytest.mark.parametrize(
    "operation",
    [
        ("apply"),
        ("undo"),
        ("both"),
    ],
)
def test_branch_with_split_unimplemented(operation):
    with pytest.raises(TypeError):
        _ = Pipeline(
            (FakeIndex(2), FakeIndex()),
            SpliterUnImplemented(operation=operation),  # type: ignore
        )
