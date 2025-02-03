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
Register Modifications
"""

from __future__ import annotations

from typing import Callable, Any, Type
import warnings

import pyearthtools.data

MODIFICATION_DICT: dict[str, Type["pyearthtools.data.modifications.Modification"]] = {}


def register_modification(name: str) -> Callable:
    """
    Register a modification for use with `@pyearthtools.data.indexes.decorators.variable_modifications`.

    Args:
        name (str):
            Name under which the modification should be registered. A warning is issued
            if this name conflicts with a preexisting modification.
    """

    def decorator(modification_class: Any):
        """Register `accessor` under `name` on `cls`"""
        if name in MODIFICATION_DICT:
            warnings.warn(
                f"Registration of modification {modification_class!r} under name {name!r} is "
                "overriding a preexisting modification with the same name.",
                pyearthtools.data.AccessorRegistrationWarning,
                stacklevel=2,
            )
        MODIFICATION_DICT[name] = modification_class

        return modification_class

    return decorator
