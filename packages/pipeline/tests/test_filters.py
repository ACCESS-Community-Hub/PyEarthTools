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


import pyearthtools.utils
import pytest
from pyearthtools.data import Index
from pyearthtools.pipeline import Operation, Pipeline, exceptions, filters, iterators

pyearthtools.utils.config.set({"pipeline.run_parallel": False})


class FakeIndex(Index):
    """Simply returns the `idx` or `override`."""

    def __init__(self, override: int | None = None):
        self._overrideValue = override
        super().__init__()

    def get(self, idx):
        return self._overrideValue or idx


class ReplaceOnKey(Operation):
    def __init__(self, **replaces):
        super().__init__(operation="apply")
        self.replaces = replaces

    def apply_func(self, sample):
        if str(sample) in self.replaces:
            return self.replaces[str(sample)]
        return sample


def test_type_filter():
    pipe = Pipeline(
        FakeIndex(), ReplaceOnKey(**{"10": "break"}), filters.TypeFilter(int), iterator=iterators.Range(0, 20)
    )

    with pytest.raises(exceptions.PipelineFilterException):
        pipe[10]

    assert len(list(pipe)) == 19
