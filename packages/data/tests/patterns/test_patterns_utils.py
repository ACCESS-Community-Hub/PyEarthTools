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

from pyearthtools.data.patterns import utils
import os
import pytest


def test_parse_path(monkeypatch):

    monkeypatch.setitem(os.environ, "SPECIAL", "fake_username")

    # Test temporary directory request
    test_path = "temp"
    result = utils.parse_root_dir(test_path)
    assert "tmp" in str(result)

    # Test variable replacement
    test_path = "/home/fictional/path/$SPECIAL/root_dir"
    result = utils.parse_root_dir(test_path)
    assert "/home/fictional/path/fake_username/root_dir" in str(result)

    # Test nonexistent variable request
    with pytest.raises(ValueError):
        test_path = "/home/fictional/path/$NONEXISTENTREALLYREALLYREALLY/root_dir"
        result = utils.parse_root_dir(test_path)
