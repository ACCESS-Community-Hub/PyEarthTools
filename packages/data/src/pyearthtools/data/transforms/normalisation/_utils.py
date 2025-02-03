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


from __future__ import annotations

CLASS_NAME_TO_TRIM = "pyearthtools.data"


def format_class_name(class_to_find: object) -> list[str]:
    """
    Format class name for use in normalisation caching

    Args:
        class_to_find (str): Class to find name for

    Returns:
        list[str]: Components of class name
    """
    class_str = str(class_to_find.__class__).split("'")[1]
    class_str = class_str.replace(CLASS_NAME_TO_TRIM, "")
    class_str_list = class_str.strip().split(".")

    if "" in class_str_list:
        class_str_list.remove("")
    return []
