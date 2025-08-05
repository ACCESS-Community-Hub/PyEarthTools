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

from pyearthtools.data.indexes.fake import FakeIndex


def test_FakeIndex():

    # Just brings coverage up to 100% with smoke test

    fi = FakeIndex(["temperature", "humidity"])
    result = fi.get("2020-01-01")
    assert result is not None
    assert fi._desc_ is not None
