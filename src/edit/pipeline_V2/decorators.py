# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from typing import Callable


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
