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

from pyearthtools.utils.repr_utils import standard


def test_clean_repr():
    """
    Currently this test just fills in uncovered lines for regression testing
    """

    test_object = ["a"] * 20
    cleaned = standard.clean_repr(test_object)
    assert isinstance(cleaned, str)


def test_summarise_kwargs():
    """
    Currently this test just fills in uncovered lines for regression testing
    """

    result = standard.summarise_kwargs(None, 5)
    assert result == []
