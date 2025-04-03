# Copyright Commonwealth of Australia, Bureau of Meteorology 2025.
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

from pyearthtools.data.operations import _aggregation


def test_aggregation(monkeypatch):
    def mock_aggregation(dataset):
        return dataset

    class MockThing:
        def over(self, *args, **kwargs):
            return mock_aggregation

        def leaving(self, *args, **kwargs):
            return mock_aggregation

    monkeypatch.setattr(_aggregation, "aggr_trans", MockThing())

    # Intended functionality
    result = _aggregation.aggregation("dataset", "aggregation", "reduction")
    assert result == "dataset"

    result = _aggregation.aggregation("a", "b", None, preserve_dims="d")
    assert result == "a"

    # Test exception handling
    with pytest.raises(ValueError):
        result = _aggregation.aggregation("a", "b")

    with pytest.raises(ValueError):
        result = _aggregation.aggregation("a", "b", "c", preserve_dims="d")
