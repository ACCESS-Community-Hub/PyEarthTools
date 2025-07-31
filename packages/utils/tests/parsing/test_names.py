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

import pyearthtools.utils.parameter
from pyearthtools.utils.parsing import names


def test_function_name():
    """
    This test just provides coverage of the function_name method
    """

    # Test a function pointer
    name = names.function_name(names.function_name)
    assert name == "pyearthtools.utils.parsing.names.function_name"

    # Test a type object
    name = names.function_name(type(names.function_name))
    assert name == "function"

    # Test a class (not an object)
    name = names.function_name(pyearthtools.utils.parameter.SingleParameter)
    assert name == "pyearthtools.utils.parameter.SingleParameter"

    class MockThing:

        def __init__(self):
            self.__module__ = "__builtin__"

        def __call__(self):
            return "Hello"

    # Test a callable object
    mt = MockThing()
    assert mt() == "Hello"
    name = names.function_name(mt)
    assert name == "MockThing"
