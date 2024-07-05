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

from edit.pipeline import Pipeline, branching, exceptions

from tests.fake_pipeline_steps import FakeIndex

edit.utils.config.set({"pipeline.run_parallel": False})

# TODO


class Unifer(branching.Unifier):
    def check_validity(self, sample):
        return 0


def test_branch_with_join_invalid():
    pipe = Pipeline((FakeIndex(1), FakeIndex(2)), branching.unify.Equality())
    assert pipe[1] == (1, 2)
    with pytest.raises(exceptions.PipelineUnificationException):
        pipe.undo(pipe[1])
