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

import pyearthtools.zoo


@pytest.mark.parametrize(
    "value, result",
    [
        # ints
        ("1", 1),
        ("100", 100),
        # float
        ("1.0", 1.0),
        ("100", 100.0),
        # bool
        ("true", True),
        ("True", True),
        ("false", False),
        ("False", False),
        # str back
        ("test_1.0", "test_1.0"),
        ("test_true", "test_true"),
        ("test_1", "test_1"),
        ("None", "None"),
    ],
)
def test_parse_str(value, result):
    assert pyearthtools.zoo.utils.parse_str(value) == result
