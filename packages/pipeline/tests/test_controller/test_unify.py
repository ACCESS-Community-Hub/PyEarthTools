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

from pyearthtools.pipeline import Pipeline, branching, exceptions

from tests.fake_pipeline_steps import FakeIndex

pyearthtools.utils.config.set({"pipeline.run_parallel": False})

# TODO


class Unifer(branching.Unifier):
    def check_validity(self, sample):
        return 0


def test_branch_with_join_invalid():
    pipe = Pipeline((FakeIndex(1), FakeIndex(2)), branching.unify.Equality())
    assert pipe[1] == (1, 2)
    with pytest.raises(exceptions.PipelineUnificationException):
        pipe.undo(pipe[1])
