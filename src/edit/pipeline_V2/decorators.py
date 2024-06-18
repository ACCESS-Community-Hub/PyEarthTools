# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from typing import Callable

from edit.pipeline_V2.exceptions import PipelineRuntimeError


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
