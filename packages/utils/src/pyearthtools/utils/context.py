# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
Contexts
"""
from __future__ import annotations
from typing import Any, Type, Callable
import logging

from contextlib import ContextDecorator


class ChangeValue(ContextDecorator):
    """
    Context Manager to change attribute of object and revert after

    Example:
        ```python
        object.attribute = 'value'
        print(object.attribute) # 'value'

        with ChangeValue(object, key = 'attribute', value = 'NewValue'):
            object.attribute = 'NewValue'
            print(object.attribute) # 'NewValue'

        print(object.attribute) # 'value'
        ```
    """

    def __init__(self, object: Any, key: str, value: Any):
        """Update Attribute of an object

        Args:
            object (Any):
                Object to update
            key (str):
                Attribute Name
            value (Any):
                Value to update `key` to

        Raises:
            AttributeError:
                If object has no attribute key
        """

        if not hasattr(object, key):
            raise AttributeError(f"{type(object)!r} has no attribute {key!r}")

        self.object = object
        self.key = key
        self.value = value

        self.original_value = getattr(object, key)

    def __enter__(self):
        setattr(self.object, self.key, self.value)
        return self.object

    def __exit__(self, *args):
        setattr(self.object, self.key, self.original_value)


class Catch(ContextDecorator):
    """
    Catch and ignore exceptions raised within scope of the context
    """

    def __init__(
        self,
        exceptions: tuple[Type[Exception]] | Type[Exception],
        *excep: Type[Exception],
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Catch exceptions occuring within context.

        Can also log exceptions if given a logger.

        Args:
            exceptions (tuple[Type[Exception]] | Type[Exception]):
                Types of exceptions to catch.
            logger (logging.Logger | None, optional):
                Logger to log exceptions to if given. Logs as debug. Defaults to None.
        """
        if not isinstance(exceptions, tuple):
            exceptions = (exceptions,)

        if excep:
            list_exceptions = list(exceptions)
            list_exceptions.extend(excep)
            exceptions = tuple(list_exceptions)  # type: ignore

        self.exceptions = exceptions
        self.logger = logger

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type in self.exceptions:
            if self.logger is not None:
                self.logger.debug(f"A {exc_type} was raised but captured. {exc_value}.")
            return True
        return False


class PrintOnError(ContextDecorator):
    """
    Print a `msg` on exception
    """

    def __init__(self, msg: str | Callable) -> None:
        """ """
        self.msg = msg

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            if isinstance(self.msg, Callable):
                self.msg = self.msg()
            print(self.msg)
        return False
