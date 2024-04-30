# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

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
