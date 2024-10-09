# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

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
