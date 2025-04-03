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

from pyearthtools.data.transforms.normalisation import _utils
import pyearthtools.data.time


def test_format_class_name():

    obj = pyearthtools.data.time.Petdt("2010-01-01")
    result = _utils.format_class_name(obj)
    assert result == ["time", "Petdt"]
