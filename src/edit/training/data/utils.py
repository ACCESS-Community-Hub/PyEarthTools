from typing import Any, Union
import xarray as xr
import datetime

import importlib
import builtins

import re

import edit.data


def get_callable(module: str):
    """
    Provide dynamic import capability

    Parameters
    ----------
        module
            String of path the module, either module or specific function/class

    Returns
    -------
        Specified module or function
    """

    try:
        return getattr(builtins, module)
    except:
        pass

    try:
        return importlib.import_module(module)
    except ModuleNotFoundError:
        module = module.split(".")
        return getattr(get_callable(".".join(module[:-1])), module[-1])


def get_class(root_module, class_name):
    if not class_name:
        return root_module
    if isinstance(class_name, str):
        class_name = class_name.split(".")
    return get_class(getattr(root_module, class_name[0]), class_name[1:])


def get_indexes(sources: dict, order: list = None):
    indexes = []

    order = order or list(sources.keys())

    for index in order:
        kwargs = sources[index]
        data_index = None

        index = re.sub(r"\[[0-9]*\]", "", index)

        try:
            data_index = get_class(edit.data, index)
        except:
            pass

        errors = []

        if not data_index:
            for alterations in [
                "__main__.",
                "",
                "edit.training.data.",
                "edit.data.",
                "edit.training.data.operations.",
                "",
            ]:
                try:
                    data_index = get_callable(alterations + index)
                except (
                    ModuleNotFoundError,
                    ImportError,
                    AttributeError,
                    ValueError,
                ) as e:
                    errors.append(e)
                    pass
                if data_index:
                    break

        if not data_index:
            raise ValueError(f"Unable to load {index!r}.\nDue to {errors}")

        if not callable(data_index):
            if hasattr(data_index, index.split(".")[-1]):
                data_index = getattr(data_index, index.split(".")[-1])
            else:
                raise TypeError(f"{index!r} is a {type(data_index)}, must be callable")
        try:
            indexes.append(data_index(**kwargs))
        except Exception as e:
            raise RuntimeError(f"Initialising {index} raised {e}")
    return indexes


def get_transforms(sources: dict, order: list = None):
    indexes = []

    order = order or list(sources.keys())

    for transform in order:
        kwargs = sources[transform]
        data_transform = None

        transform = re.sub(r"\[[0-9]*\]", "", transform)

        try:
            data_transform = get_class(edit.data.transform, transform)
        except:
            pass

        if not data_transform:
            for alterations in ["__main__.", "", "edit.data.transform", "edit.data."]:
                try:
                    data_transform = get_callable(alterations + transform)
                except (ModuleNotFoundError, ImportError, AttributeError, ValueError):
                    pass
                if data_transform:
                    break

        if not data_transform:
            raise ValueError(f"Unable to load {transform!r}")

        if not callable(data_transform):
            if hasattr(data_transform, transform.split(".")[-1]):
                data_transform = getattr(data_transform, transform.split(".")[-1])
            else:
                raise TypeError(
                    f"{transform!r} is a {type(data_transform)}, must be callable"
                )
        try:
            indexes.append(data_transform(**kwargs))
        except Exception as e:
            raise RuntimeError(f"Initialising {transform} raised {e}")
    return edit.data.transform.TransformCollection(indexes)
