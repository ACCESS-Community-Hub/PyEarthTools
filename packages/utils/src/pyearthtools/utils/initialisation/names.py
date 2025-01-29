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


from typing import Callable
import types


def function_name(object: Callable) -> str:
    """
    Get Function Name of step

    Args:
        object (Callable): Callable to get name of

    Returns:
        str: Module path to Callable
    """
    if isinstance(object, type):
        return str(object).split("'")[1]

    module = object.__module__

    if isinstance(object, types.FunctionType):
        name = object.__name__
    else:
        name = object.__class__.__name__

    str_name = str(name)
    if "<locals>" in str_name:
        return str_name.split("'")[1].split("<locals>")[0].removesuffix(".")

    if module is not None and module != "__builtin__":
        name = module + "." + str(name)
    return name
