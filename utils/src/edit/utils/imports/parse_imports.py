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
import re
from typing import Any, Callable
from pathlib import Path
import warnings

import yaml


DEFAULT_CHECK_LOCATIONS = [
    "",
    "__main__.",
    "edit.",
]


def dynamic_import(object_path: str) -> Callable:
    """
    Provide dynamic import capability

    Args:
        object_path (str): Path to import

    Raises:
        (ImportError, ModuleNotFoundError): If cannot be imported

    Returns:
        (Callable): Imported objects
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
        object_path = object_path.split(".")
        return getattr(dynamic_import(".".join(object_path[:-1])), object_path[-1])
    except ValueError as e:
        raise ModuleNotFoundError("End of module definition reached")


def get_class(root_module: object, class_name: str | list[str]) -> object:
    """
    Get class from module

    Allows `class_name` to be a path, e.g. `sub_module.class`,
    will resolve through recursive decent.

    Args:
        root_module (object):
            Object to find class in
        class_name (str list[str]):
            Class name, can be multi level. Or
            use list as levels.

    Raises:
        AttributeError: If class could not be found.

    Returns:
        (object):
            Discovered class underneath `root_module`
    """
    if not class_name:
        return root_module

    if isinstance(class_name, str):
        class_name = class_name.split(".")
    if not hasattr(root_module, class_name[0]):
        raise AttributeError(f"{root_module} has no attribute {class_name[0]}")

    try:
        return get_class(getattr(root_module, class_name[0]), class_name[1:])
    except AttributeError:
        pass
    raise AttributeError(f"Could not find {'.'.join(class_name)} underneath {root_module}")


def get_items(
    sources: dict[str, Any],
    order: list[str] | None = None,
    first_check: Callable | Any = None,
    import_locations: list[str] = DEFAULT_CHECK_LOCATIONS,
    callback: Callable | None = None,
    resolving_path: str | Path | None = None,
):
    """Load classes or other callables and initalise them from a dictionary.

    !!! tip "Path Tip"
        A path to the class doesn't always have to be specified, `import_locations` allows prefixs to be set

    !!! tip "Multiple Tip"
        If two or more of the same class are wanted, add '[a-Z 0-9]', to distinguish the key, this will be removed before import


    Args:
        sources (dict[str, Any]):
            Dictionary specifying items to load and keyword arguments to pass.
            Can be string pointing to yaml file
        order (list[str], optional):
            Override for order to load them in. Defaults to None.
        first_check (Callable | Any, optional):
            Class or module to check first. Defaults to None.
        import_locations (list[str], optional):
            Locations to import from. Defaults to DEFAULT_CHECK_LOCATIONS
        callback (Callable, optional):
            Callback function if source appears to be another file. Passes only the file path,
            Use `functools.partial` if needed.
            If None, call this function passing all args except `order`.
            Defaults to None.
        resolving_path (str | Path, optional):
            Location to resolve relative paths if 'file' in sources.
            Defaults to None.

    Raises:
        ValueError:
            If an error occurs importing the item
        TypeError:
            If an invalid type was imported
        RuntimeError:
            If an error occurs initialising the items

    Returns:
        (list[Any]):
            Imported and Initialised objects from the configuration

    """
    items = []

    resolving_path = resolving_path if resolving_path is None else Path(resolving_path)

    ## Open if is file, assuming yaml
    if isinstance(sources, (Path, str)) and Path(sources).exists():
        with open(sources, "r") as file:
            sources = yaml.safe_load(file)

    ## Get order
    order = order or list(sources.keys())

    ## Go through each item
    for item in order:
        ## Get init args
        init_args = sources[item]
        imported_item = None

        errors = []

        ## Remove number marks
        item = re.sub(r"\[.*\]", "", item)

        ## Try from class
        if first_check:
            try:
                imported_item = get_class(first_check, item)
            except AttributeError:
                pass

        ## If not found, try importing from set locations
        if not imported_item:
            for alterations in import_locations:
                try:
                    imported_item = dynamic_import(alterations + item)
                except (
                    ModuleNotFoundError,
                    ImportError,
                    AttributeError,
                    ValueError,
                ) as e:
                    e.add_note(f"Looking in {import_locations}")
                    errors.append(e)
                    pass
                ## If found break out
                if imported_item:
                    break

        ## If still not found, check if its a string / Path
        if not imported_item:
            if isinstance(item, str) and "file" in item:
                item = init_args

                try:
                    path_item = Path(item)
                    if resolving_path is not None and not path_item.is_absolute():
                        path_item = (resolving_path / path_item).resolve()

                    if path_item.exists():
                        if callback is None:
                            new_items = get_items(
                                path_item,
                                first_check=first_check,
                                import_locations=import_locations,
                                resolving_path=resolving_path,
                            )
                        else:
                            new_items = callback(path_item)

                        new_items = new_items if isinstance(new_items, (list, tuple)) else [new_items]
                        for new_i in new_items:
                            items.append(new_i)

                        ## Leave for loop if loaded from file
                        continue
                    else:
                        warnings.warn(
                            f"item could be a file, but does not exists.\nCould not find {item!r}", UserWarning
                        )

                except Exception as e:
                    warnings.warn(f"item could be a file, but could not be loaded. {item!r}", UserWarning)
                    errors.append(e)

                if not imported_item:
                    errors_str = "\n".join([str(e) for e in errors])
                    raise ValueError(
                        f"Unable to load {item!r} which looked like a file.\nThe following errors occured: {errors_str}"
                    )

        ## Still not found, raise error
        if not imported_item:
            raise ExceptionGroup(f"Unable to load {item!r} from {import_locations}.", errors)
            # raise ValueError(f"Unable to load {item!r} from {import_locations}.\nThe following errors occured: {errors_str}")

        ## Get callable form of item
        if not callable(imported_item):
            if hasattr(imported_item, item.split(".")[-1]):
                imported_item = getattr(imported_item, item.split(".")[-1])
            else:
                raise TypeError(f"{item!r} is a {type(imported_item)}, must be callable")

        ## Initialise object
        try:
            if isinstance(init_args, (tuple, list)):
                items.append(imported_item(*init_args))
            elif isinstance(init_args, dict):
                items.append(imported_item(**init_args))
            else:
                items.append(imported_item(init_args))

        except Exception as e:
            raise RuntimeError(f"Error occurred initialising {item}") from e

    if len(items) == 1:
        return items[0]
    return items
