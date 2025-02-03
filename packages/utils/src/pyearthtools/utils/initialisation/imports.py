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
Load Classes from dictionary or strings
"""


from __future__ import annotations
import builtins

import importlib

from types import ModuleType
from typing import Callable


def dynamic_import(object_path: str) -> Callable | ModuleType:
    """
    Provide dynamic import capability

    Args:
        object_path (str): Path to import

    Raises:
        (ImportError, ModuleNotFoundError): If cannot be imported

    Returns:
        (Callable | ModuleType): Imported objects
    """
    try:
        return getattr(builtins, object_path)
    except AttributeError:
        pass

    if not object_path:
        raise ImportError(f"object_path cannot be empty")
    try:
        return importlib.import_module(object_path)
    except ModuleNotFoundError:
        object_path_list = object_path.split(".")
        return getattr(dynamic_import(".".join(object_path_list[:-1])), object_path_list[-1])
    except ValueError as e:
        raise ModuleNotFoundError("End of module definition reached")
