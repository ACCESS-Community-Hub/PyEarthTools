# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
Register Modifications
"""

from __future__ import annotations

from typing import Callable, Any, Type
import warnings

import edit.data

MODIFICATION_DICT: dict[str, Type["edit.data.modifications.Modification"]] = {}


def register_modification(name: str) -> Callable:
    """
    Register a modification for use with `@edit.data.indexes.decorators.variable_modifications`.

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
                edit.data.AccessorRegistrationWarning,
                stacklevel=2,
            )
        MODIFICATION_DICT[name] = modification_class

        return modification_class

    return decorator
