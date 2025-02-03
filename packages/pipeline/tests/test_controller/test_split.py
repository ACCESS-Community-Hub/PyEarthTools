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

from pyearthtools.pipeline import Pipeline, branching

from tests.fake_pipeline_steps import FakeIndex

pyearthtools.utils.config.set({"pipeline.run_parallel": False})


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
