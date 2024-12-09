# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations

import functools
from typing import Any, Callable

import warnings


class classproperty(property):
    """Set a method available as a property on the class"""

    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)  # type: ignore


def invert_dictionary_list(dictionary: dict) -> dict:
    return_dict = {}
    for key, value in dictionary.items():
        for item in value:
            return_dict[item] = key
    return return_dict


def alias_arguments(**aliases: str | list[str]) -> Callable:
    """
    Setup aliases for parameters

    Args:
        **aliases (str | list[str]):
            Dictionary pair, of true name to aliases

            Values can be either str or list of strings

    Examples:
        >>> @alias_arguments(response = ['rep', 'answer'])
            def function(response):
                return response
        >>> function('yes')
        ... 'yes'
        >>> function(rep = 'maybe')
        ... 'maybe'
        >>> function(hello = 'maybe')
        ... # An Error is raised

    """

    def internal_function(func: Callable) -> Callable:
        # Force all aliases to be list
        for k, v in aliases.items():
            if isinstance(v, (list, tuple)):
                continue
            elif isinstance(v, str):
                aliases[k] = [v]
                continue
            raise ValueError(f"Invalid type for alias.'{k}':{v}")

        flipped_aliases = invert_dictionary_list(aliases)

        @functools.wraps(func)  # type: ignore
        def wrapper(*args, **kwargs):
            new_kwargs = {}

            for k, v in kwargs.items():
                if k in flipped_aliases:
                    k = flipped_aliases[k]
                if k in new_kwargs:
                    if v is None:
                        continue
                    else:
                        raise ValueError(
                            f"Two keyword arguments for {k!r} have been passed due to alias resolving. Cannot parse.\nRecieved: {kwargs!r}."
                        )
                new_kwargs[k] = v
            return func(*args, **new_kwargs)

        return wrapper

    return internal_function


def BackwardsCompatibility(new_func: Callable[[Any], Any]):
    """
    Allows for the renaming of a functionality, and subsequent backwards compatilbility.

    Will warn about deprecation on use.

    Args:
        new_func:
            New function to point to instead

    Example:
        >>> def new_func(): # New function name
            ...
        >>> @BackwardsCompatibility(new_func)
        >>> def old_func(*a, **kw):
            ...

    """

    @functools.wraps(new_func)
    def decorator(func):
        @functools.wraps(new_func)
        def wrapped(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} has been removed in favour of {new_func.__name__}, please switch over.",
                DeprecationWarning,
            )
            return new_func(*args, **kwargs)

        return wrapped

    return decorator
