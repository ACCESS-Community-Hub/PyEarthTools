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


"""
Data processing tools for use by the [Tesselator][pyearthtools.utils.data.Tesselator]
"""
DEFAULT_FORMAT_SUBSET: str = "...HW"

DEFAULT_FORMAT_PATCH_ORGANISE: str = "P...HW"
DEFAULT_FORMAT_PATCH: str = "RP...HW"
DEFAULT_FORMAT_PATCH_AFTER: str = "...HW"

from . import patches, reorder, subset
