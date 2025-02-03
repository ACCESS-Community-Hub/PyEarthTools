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

from pyearthtools.pipeline.exceptions import PipelineRuntimeError


def potentialabstractmethod(func: Callable):
    """A decorator indicating potential abstract methods.

    The class using this may then check the function for the property
    `__ispotentialabstractmethod__` to determine if it was implemented.

    Usage:

        class C():
            @potentialabstractmethod
            def my_potential_abstract_method(self, ...):
                ...
    """
    setattr(func, "__ispotentialabstractmethod__", True)
    return func


class PotentialABC:
    """
    Check if `potentialabstractmethod` are needed and if so are implemented.
    """

    def check_abstractions(self, required_methods: list[str]):
        """
        Check `potentialabstractmethod`'s

        Args:
            required_methods (list[str]):
                List of method to check

        Raises:
            PipelineRuntimeError:
                If method was not implemented.
        """
        for method in required_methods:
            if getattr(getattr(self, method), "__ispotentialabstractmethod__", False):
                raise PipelineRuntimeError(
                    f"Can't instantiate {self.__class__.__qualname__!s} as `{method}` is not implemented and is expected."
                )
