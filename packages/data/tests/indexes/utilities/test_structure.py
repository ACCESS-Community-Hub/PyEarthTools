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
import tempfile

from pyearthtools.data.indexes.utilities import structure

filter_tests = [
    # Filter out a name
    {"names": ["temperature", "wind", "wombats"], "disallowed": "wombats", "expected": ["temperature", "wind"]},
    # Filter out something that's not there
    {"names": ["temperature", "wind"], "disallowed": "wombats", "expected": ["temperature", "wind"]},
]


@pytest.mark.parametrize("test_dictionary", filter_tests)
def test_filter_disallowed(test_dictionary):
    td = test_dictionary

    result = structure.filter_disallowed(td["names"], td["disallowed"])
    assert result == td["expected"]


def test_get_structure():

    with tempfile.TemporaryDirectory() as location:

        # TODO: Create some temporary files in here and test
        # the disallow functionality also
        structure.get_structure(location, [])


def test_clean_structure():

    structure.clean_structure(
        {"names": ["temperature", "wind", "wombats"], "disallowed": "wombats", "expected": ["temperature", "wind"]}
    )
