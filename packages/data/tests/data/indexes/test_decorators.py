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

from pyearthtools.data.indexes import decorators


@decorators.alias_arguments(value=["v", "val"])
@decorators.check_arguments(value=[1, 2])
def fake_function(value):
    return value


@pytest.mark.parametrize(
    "key, value, error",
    [
        ("value", 1, False),
        ("val", 1, False),
        ("why", 1, True),
        ("value", 1, False),
        ("value", 2, False),
        ("value", 0, True),
        ("value", "Test", True),
    ],
)
def test_fake_function(key, value, error: bool):
    if error:
        with pytest.raises(Exception):
            assert fake_function(**{key: value}) == value
    else:
        assert fake_function(**{key: value}) == value
