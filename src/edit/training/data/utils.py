"""
Utilty functions for edit.training.data
"""

from typing import Any, Callable, Union
import xarray as xr

import importlib
import builtins

import re

import edit.data
from edit.training.data import templates


def get_callable(module: str) -> Callable:
    """Provide dynamic import capability.

    Allows a string name of a module or class to be given, and an imported callable returned

    Args:
        module (str): 
            String of path the module, either module or specific function/class

    Returns:
        (Callable): 
            Specified module or function
    
    Examples:
        >>> get_callable('Exception')
        Exception
        >>> get_callable('edit.training')
        <module 'edit.training' from ... >
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
    except ValueError as e:
        raise ModuleNotFoundError("End of module definition reached")



def get_class(root_module, class_name):
    if not class_name:
        return root_module
    if isinstance(class_name, str):
        class_name = class_name.split(".")
    return get_class(getattr(root_module, class_name[0]), class_name[1:])


def get_pipeline(sources: dict, order: list = None) -> list[Any]:
    """Load [pipeline steps][edit.training] and initalise them from a dictionary.

    !!! tip "Path Tip"
        A path to the class doesn't always have to be specified, the below are automatically tried.

        - `__main__.`
        - `edit.training.data.`
        - `edit.data.`
        - `edit.training.data.operations.`

    !!! tip "Multiple Tip"
        If two or more of the same [DataStep][edit.training.data.DataStep] are wanted, add '[NUMBER]', to distinguish the key, this will be removed before import

    !!! Warning
        If `index` is not provided to [edit.training.data][edit.training.data] classes, they will not be fully initialised, due to [SequentialIterator][edit.training.data.sequential.SequentialIterator].

        Suggested to instead use [sequential.from_dict][edit.training.data.sequential.from_dict].

    Args:
        sources (dict): 
            Dictionary specifying pipeline steps to load and keyword arguments to pass
        order (list, optional): 
            Override for order to load them in. Defaults to None.

    Raises:
        ValueError: 
            If an error occurs importing the step
        TypeError: 
            If an invalid type was imported
        RuntimeError: 
            If an error occurs initialising the steps

    Returns:
        (list[Any]): 
            Imported and Initialised objects from the configuration

    Examples:
        >>> get_pipeline(sources = {'filters.DropNan':{}, 'reshape.Squish': {'axis': 1}})
        #Somewhat loaded pipeline containing those two steps, but waiting on an index, as it wasn't given
        [Sequential Iterator for DropNan waiting on an index., Sequential Iterator for Squish waiting on an index.]
    """    
    indexes = []
    if isinstance(sources, templates.DataStep):
        return sources

    order = order or list(sources.keys())

    for index in order:
        init_args = sources[index]
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
            if isinstance(init_args, list):
                indexes.append(data_index(*init_args))
            elif isinstance(init_args, dict):
                indexes.append(data_index(**init_args)) 
        except Exception as e:
            raise RuntimeError(f"Error occurred initialising {index}") from e
    return indexes


def get_transforms(sources: dict, order: list = None) -> list[edit.data.Transform]:
    """Load [Transforms][edit.data.transform] and initalise them from a dictionary.

    !!! tip "Path Tip"
        A path to the class doesn't always have to be specified, the below are automatically tried.

        - `__main__.`
        - `edit.data.transform.`
        - `edit.data.`
        - `edit.training.data.operations.transforms.`

    !!! tip "Multiple Tip"
        If two or more of the same [Transform][edit.data.transform] are wanted, add '[NUMBER]', to distiguish the key, this will be removed before import

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
            If an error occurs intialising the transforms

    Returns:
        (list[edit.data.Transform]): 
            Imported and Initalised Transforms from the configuration

    Examples:
        >>> get_transforms(sources = {'region.lookup':{'key': 'Adelaide'}})
        Transform Collection:
        BoundingCut                   Cut Dataset to Adelaide region
    """      

    transforms = []

    if isinstance(sources, edit.data.Transform):
        return sources

    elif isinstance(sources, dict):
        order = order or list(sources.keys())

        for transform in order:
            init_args = sources[transform]
            data_transform = None

            transform = re.sub(r"\[[0-9]*\]", "", transform)

            try:
                data_transform = get_class(edit.data.transform, transform)
            except:
                pass

            if not data_transform:
                for alterations in ["__main__.", "", "edit.data.transform.", "edit.data." , "edit.training.data.operations.transforms."]:
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
                if isinstance(init_args, list):
                    transforms.append(data_transform(*init_args))
                elif isinstance(init_args, dict):
                    transforms.append(data_transform(**init_args))            
            except Exception as e:
                raise RuntimeError(f"Error occured initialising {data_transform}") from e
        return edit.data.transform.TransformCollection(transforms)
    return edit.data.transform.TransformCollection(sources)