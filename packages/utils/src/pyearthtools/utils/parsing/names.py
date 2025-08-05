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


import types
from typing import Callable


def function_name(anobject: Callable) -> str:
    """
    Get Function Name of step

    Args:
        object (Callable): Callable to get name of

    Returns:
        str: Module path to Callable
    """
    if isinstance(anobject, type):
        return str(anobject).split("'")[1]

    module = anobject.__module__

    if isinstance(anobject, types.FunctionType):
        name = anobject.__name__
    else:
        name = anobject.__class__.__name__

    # Not covered by testing. Presumably here for a reason. Leaving for posterity.
    # if "<locals>" in str_name:
    #     return str_name.split("'")[1].split("<locals>")[0].removesuffix(".")

    if module is not None and module != "__builtin__":
        name = module + "." + str(name)
    return name
