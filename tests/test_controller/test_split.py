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

from edit.pipeline_V2 import Pipeline, branching

from tests.fake_pipeline_steps import *


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
        pipe = Pipeline(
            (FakeIndex(2), FakeIndex()),
            SpliterUnImplemented(operation=operation),
        )
