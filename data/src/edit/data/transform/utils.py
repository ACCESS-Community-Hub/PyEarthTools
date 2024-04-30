# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

"""
Transform Utility Functions
"""

from __future__ import annotations

import xarray as xr
from typing import Any, Callable
from pathlib import Path

import edit.data
from edit.utils.imports import get_items

DEFAULT_TRANSFORM_LOCATIONS = [
    "__main__.",
    "",
    "edit.data.transform.",
    "edit.data.",
]


def get_transforms(
    sources: dict, order: list | None = None
) -> edit.data.transform.Transform | edit.data.transform.TransformCollection:
    """Load [Transforms][edit.data.transform] and initialise them from a dictionary.

    !!! tip "Path Tip"
        A path to the class doesn't always have to be specified, the below are automatically tried.

        - `__main__.`
        - `edit.data.transform.`
        - `edit.data.`

    !!! tip "Multiple Tip"
        If two or more of the same [Transform][edit.data.transform] are wanted, add '[NUMBER]', to distinguish the key, this will be removed before import

    Args:
        sources (dict):
            Dictionary specifying transforms to load and keyword arguments to pass
        order (list, optional):
            Override for order to load them in. Defaults to None.

    Raises:
        ValueError:
            If an error occurs importing the transform
        TypeError:
            If an invalid type was imported
        RuntimeError:
            If an error occurs initialising the transforms

    Returns:
        (edit.data.transform.Transform | edit.data.transform.TransformCollection):
            Imported and Initialised Transforms from the configuration

    Examples:
        >>> get_transforms(sources = {'region.lookup':{'key': 'Adelaide'}})
        Transform Collection:
        BoundingCut                   Cut Dataset to Adelaide region
    """
    transforms = []

    if isinstance(sources, edit.data.transform.Transform):
        return sources

    transforms = get_items(
        sources,
        order,
        edit.data.transform,
        import_locations=DEFAULT_TRANSFORM_LOCATIONS,
    )
    return edit.data.transform.TransformCollection(transforms)


def function_name(object: Callable) -> str:
    """
    Get Function Name of Transform

    Args:
        object (Callable): Callable to get name of

    Returns:
        str: Module path to Callable
    """
    module = object.__module__
    name = object.__class__

    if "<locals>" in str(name):
        return str(name).split("'")[1].split("<locals>")[0].removesuffix(".")

    str_name = object.__class__.__name__
    if module is not None and module != "__builtin__":
        str_name = module + "." + str_name
    return str_name


def parse_transforms(
    transforms: dict | list | tuple | edit.data.transform.Transform | edit.data.transform.TransformCollection,
) -> dict:
    """
    Convert transforms to dictionary which they can be initalised from
    """
    if isinstance(transforms, dict):
        return transforms
    transform_dict: dict[str, dict] = {}

    transforms = edit.data.transform.TransformCollection(transforms)

    for transf in transforms:
        if not hasattr(transf, "_info_") or transf._info_ is NotImplemented:
            continue

        i = 0
        name = function_name(transf)
        new_name = str(name) + "[{i}]"

        while name in transform_dict:
            i += 1
            name = str(new_name).format(i=i)

        transform_dict[name] = transf._info_

    return transform_dict


def parse_dataset(value: str | Path | Any) -> Any:
    """
    Attempt to load dataset if value is str,
    If fails in any way, continue returning initial value
    """
    if isinstance(value, (str, Path)):
        try:
            if Path(value).exists():
                return xr.open_dataset(value)
        except Exception:
            pass
    return value


# def guess_coordinate_name(dataset) -> dict:
#     pass
# TODO Create ^
