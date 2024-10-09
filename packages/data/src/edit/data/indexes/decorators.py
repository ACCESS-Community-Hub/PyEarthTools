# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
Decorators for use by `edit.data.indexes`
"""

from __future__ import annotations

import functools
import warnings

import inspect
from pathlib import Path
from typing import Any

from edit.data.indexes.utilities import spellcheck, open_static
from edit.utils.decorators import alias_arguments

from edit.data.modifications import variable_modifications

__all__ = [
    "alias_arguments",
    "check_arguments",
    "deprecated_arguments",
    "variable_modifications",
]


def _check_required_arguments(default: dict[str, inspect.Parameter], kwargs: dict, function_object: object):
    """
    Check if all required arguments have been given,

    If an argument in `default` is not in `kwargs` and doesn't have
    a default value and is not a * / ** variable, raise Error


    Args:
        default (dict[str, inspect.Parameter]):
            Parameter dictionary, from inspect.signature
        kwargs (dict):
            Kwargs given to function
        function_object (object):
            Object for error usage.

    Raises:
        TypeError:
            If missing required argument
    """

    not_required = [
        inspect._ParameterKind.VAR_KEYWORD,
        inspect._ParameterKind.VAR_POSITIONAL,
    ]

    for key in set(default.keys()).difference(set(kwargs.keys())):
        value = default[key]
        if value.default == inspect._empty and value.kind not in not_required:
            raise TypeError(f"{function_object} missing 1 required argument: {key!r}")


def _get_default_arguments(signature: inspect.Signature):
    """Get default arguments from a `inspect.Signature`"""
    ## Get all params
    function_arguments = list(signature.parameters)

    default_values = {}
    for key in function_arguments:
        value = signature.parameters[key]
        if value.kind not in [
            inspect._ParameterKind.VAR_KEYWORD,
            inspect._ParameterKind.VAR_POSITIONAL,
        ]:
            default_values[key] = value.default

    return default_values


def _is_accepted_file(filename: str) -> bool:
    if isinstance(filename, str):
        extensions = [".valid", ".struc"]
        return Path(filename).suffix in extensions
    return False


def _check_structure(structure: str | Path | dict[str, Any], arguments: dict[str, list[str]]) -> dict[str, Any]:
    """Check structure of arguments given a `.struc` file

    Args:
        structure (str | Path | dict[str, Any]):
            Either structure of class path to
        arguments (dict[str, list[str]]):
            Arguments to check

    Raises:
        TypeError:
            If structure is not a dict or path to file of dict
        TypeError:
            If structure runs out of layers to check
        InvalidIndexError:
            If argument value is invalid

    Returns:
        (dict[str, Any]):
            Checked arguments.
            This will remove any `spellcheck.VariableDefault`
    """

    ## Load structure from file if is Path or str

    if isinstance(structure, (Path, str)):
        opened_file = open_static(structure)
        if not isinstance(opened_file, dict):
            raise TypeError(f"Opened structure file is not a dictionary. {structure} is of type {type(opened_file)}")
        structure_dict = opened_file
    elif isinstance(structure, dict):
        structure_dict = structure
    else:
        raise TypeError(f"Cannot parse structure of type {type(structure)}")

    ## Get order to assess arguments in
    order = structure_dict.pop("order", list(arguments.keys()))

    ## Check arguments
    for key in order:
        if key not in arguments:
            raise KeyError(f"Could not find value for {key!r} in given arguments. {arguments}")
        value = arguments[key]

        ## Get valid arguments
        ## If dict, all keys represent continuing tree
        if isinstance(structure_dict, dict):
            valid_args = list(structure_dict.keys())
        elif isinstance(structure_dict, (list, tuple)):
            valid_args = structure_dict
        elif isinstance(structure_dict, str):
            valid_args = [structure_dict]
        else:
            raise TypeError(f"Structure has been exhausted, too many arguments were specified. Args: {list(order)}")

        ## Check argument
        arguments[key] = value = spellcheck.check_prompt(value, valid_args, name=key)

        ## Continue through structure
        if isinstance(structure_dict, dict):
            if isinstance(value, (list, tuple)):  # Combine possible values
                new_struct = []
                for v in value:
                    s_v = structure_dict[v]
                    if not new_struct:
                        new_struct = s_v
                        continue
                    new_struct = list(set(new_struct).intersection(s_v if isinstance(s_v, (tuple, list)) else [s_v]))

                if all(map(lambda x: isinstance(x, dict), new_struct)):
                    dict_struc = {}
                    for s in new_struct:
                        dict_struc.update(**s)
                else:
                    new_struct = list(set(new_struct))

                structure_dict = new_struct

            else:  # Just use next value
                structure_dict = structure_dict[value]
        else:
            structure_dict = None

    return arguments


def check_arguments(
    struc: str | Path | dict[str, Any] | None = None,
    **valid_arguments: list[Any] | tuple[Any, ...] | str,
):
    """
    Check Arguments before passing to function,

    If arguments and true arguments are a string, will attempt to find nearby spellings

    Args:
        struc (str | Path | dict, optional):
            Structure to check arguments with.
            A structure is dictionary as a tree with each layer referencing an argument.
            Can contain an 'order' key to specify the order.
            Defaults to None
        **valid_arguments (Any):
            Dictionary pair, of parameter name and safe values

    !!! Help "File Stored Arguments"
        Both `struc` and any `valid_argument` can point to a file either `.struc` or `.valid` respectively.
        However, this file will be found using the `importlib.resources.as_file`, so must point to class path.

    !!! Tip "Dynamic List"
        The path name to a '.valid' can contain `{}` with keys of other initialisation arguments, but those must be a str

    Examples:
        >>> @check_arguments(response = ['yes', 'no'])
            def function(response):
                return response
        >>> function('yes')
        ... 'yes'
        >>> function('maybe')
        ... # An Error is Raised

        >>> @check_arguments(variable = 'module.submodule.{argument}.valid')
            def func(variable, argument = 'default'):
                return variable

    """

    def internal_function(func):
        ## Get function signature
        signature = inspect.signature(func)
        ## Get all params
        function_arguments = list(signature.parameters)
        ## Get all default values
        parameter_values = {key: signature.parameters[key] for key in function_arguments}
        default_values = _get_default_arguments(signature)

        ## Parse given valid arguments
        for k, v in valid_arguments.items():
            if k not in function_arguments:
                raise KeyError(f"{k!r} not in function signature")

            if isinstance(v, (list, tuple)):
                continue
            if isinstance(v, str):
                if _is_accepted_file(v):
                    valid_arguments[k] = v
                    continue
                valid_arguments[k] = [v]
                continue
            raise ValueError(f"Cannot parse valid arguments list for '{k}':{v}")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            ## Convert args to kwargs
            for i, arg in enumerate(args):
                if function_arguments[i] in kwargs:
                    raise TypeError(f"{func.__qualname__} got multiple values for argument {function_arguments[i]!r}")
                kwargs[function_arguments[i]] = arg

            ## Ensure all required args were given
            _check_required_arguments(
                parameter_values,
                kwargs,
                function_object=str(func.__qualname__).split(".")[0],
            )

            ## Get all non given kwargs with default values so that formatting of .valids can be done
            kwargs.update(
                {key: default_values[key] for key in set(default_values.keys()).difference(set(kwargs.keys()))}
            )

            ## Check structure
            if struc is not None:
                kwargs = _check_structure(struc, kwargs)

            ## Check all kwargs where valid arguments have been given
            for k, valid_arg in valid_arguments.items():
                ## Skip items which have valid args but were not given
                if k not in kwargs:
                    continue

                ## Load from .valid file
                if isinstance(valid_arg, str) and ".valid" in valid_arg:
                    try:
                        class_path = ".".join(valid_arg.split(".")[:-2]).format(**kwargs)
                        file_path = ".".join(valid_arg.split(".")[-2:]).format(**kwargs)
                    except KeyError as e:
                        raise KeyError(f"Formatting {valid_arg!r} raised an error") from e

                    valid_arg = open_static(class_path, file_path)

                if isinstance(kwargs[k], (tuple, list)):
                    for item in kwargs[k]:
                        spellcheck.check_prompt(item, valid_arg, k)
                else:
                    kwargs[k] = spellcheck.check_prompt(kwargs[k], valid_arg, k)

            return func(**kwargs)

        return wrapper

    return internal_function


def deprecated_arguments(
    deprecation: dict[str, str | None] | str | None = None,
    *arg_deprecations: str,
    **extra_deprecations: str | None,
):
    """
    Warn a user if they attempt to use a deprecated argument, and remove it from the call.

    If given a dictionary, the key is the arg, and the value the warning.
    """

    if deprecation is None:
        deprecation = {}
    elif isinstance(deprecation, str):
        deprecation = {deprecation: None}

    deprecation.update({k: None for k in arg_deprecations})
    deprecation.update(extra_deprecations)

    def internal_func(func):
        @functools.wraps(func)
        def warn_on_deprecated(*args, **kwargs):
            for depr in deprecation.keys():
                if depr in kwargs:
                    warning = (
                        deprecation[depr]
                        or f"{depr!r} is a deprecated keyword, and may raise an error in future versions."
                    )
                    warnings.warn(warning, DeprecationWarning)
                    kwargs.pop(depr)

            return func(*args, **kwargs)

        return warn_on_deprecated

    return internal_func
