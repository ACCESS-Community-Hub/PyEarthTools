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


import pytest
from pyearthtools.pipeline import Operation, Pipeline

from pyearthtools.data import Index


class FakeIndex(Index):
    def get(self, idx):
        return idx


class EmptyOperation(Operation):
    def __init__(self):
        super().__init__()

    def apply_func(self, sample):
        return sample

    def undo_func(self, sample):
        return sample


class AllowsOnlyTuples(Operation):
    def __init__(self, split, types):
        super().__init__(split_tuples=split, recognised_types=types)

    def apply_func(self, sample):
        return sample

    def undo_func(self, sample):
        return sample


@pytest.mark.parametrize(
    "split, type",
    [
        (False, tuple),
        (True, int),
    ],
)
def test_recognised_types_success(split, type):
    pipe = Pipeline(FakeIndex(), (EmptyOperation(), EmptyOperation()), AllowsOnlyTuples(split, type))
    assert pipe[1] == (1, 1)


@pytest.mark.parametrize(
    "split, type",
    [
        (True, tuple),
        (False, int),
    ],
)
def test_recognised_types_fails(split, type):
    pipe = Pipeline(FakeIndex(), (EmptyOperation(), EmptyOperation()), AllowsOnlyTuples(split, type))
    with pytest.raises(TypeError):
        assert pipe[1] == (1, 1)
