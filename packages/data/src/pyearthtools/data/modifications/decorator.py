# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Variable Modification decorator
"""

from __future__ import annotations

import inspect
import re
import json

import functools
from typing import Callable, Any, Optional, Type, Union
import xarray as xr

from pyearthtools.data.indexes import TimeDataIndex

from pyearthtools.data.transforms.transform import Transform
from pyearthtools.data.modifications.modification import Modification

from pyearthtools.data.modifications.register import MODIFICATION_DICT


__all__ = ["variable_modifications", "Modification"]


def _args_to_kwargs(func, args: tuple[Any, ...]) -> dict[str, Any]:
    """
    Convert args from function into kwargs
    """
    param_names = list(inspect.signature(func).parameters)
    return {param_names[i]: args[i] for i in range(len(args))}


def _default_params(func) -> dict[str, Any]:
    """
    Get default parameters
    """
    parameters = inspect.signature(func).parameters
    return {key: val.default for key, val in parameters.items() if not val.default == inspect._empty}


def _update_variables(
    func,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    variable_keyword: str,
    new_variables: Any,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """
    Update variables when either in `args` or `kwargs`
    """
    param_names = list(inspect.signature(func).parameters)[: len(args)]
    L_args = list(args)

    if isinstance(new_variables, list) and len(new_variables) == 1:
        new_variables = new_variables[0]

    if variable_keyword in param_names:
        L_args[param_names.index(variable_keyword)] = new_variables
        return tuple(L_args), kwargs
    kwargs[variable_keyword] = new_variables
    return tuple(L_args), kwargs


class VariableModification:
    _variable: str
    _modification_class: Union[str, None] = None
    _modification_dict: Union[dict[str, Any], None] = None

    def __init__(self, specification: str | dict[str, Any]):
        if isinstance(specification, str):
            self._from_str(specification)
        elif isinstance(specification, dict):
            self._from_dict(specification)
        else:
            raise TypeError(f"Cannot parse variable specification: {specification!r}.")

    def _from_str(self, specification: str):
        reg_exp = r"!(.+):(.+)"
        matches: list[tuple[str, str]] = re.findall(reg_exp, str(specification))

        if not matches:
            self._variable = specification
            return

        mod_info = matches[0][0].replace("]", "")
        if "[" not in mod_info:
            mod_info += "["
        mod_type, mod_kwargs = mod_info.split("[", 2)

        if mod_type not in MODIFICATION_DICT:
            raise ValueError(
                f"Cannot find modification {mod_type!r}, expected one of {list(MODIFICATION_DICT.keys())}."
            )

        if mod_kwargs:
            try:
                mod_kwargs = ",".join(
                    '"' + k + '":' + v for k, v in map(lambda x: x.split(":"), mod_kwargs.split(","))
                ).replace("'", '"')
                mod_kwargs = json.loads("{" + mod_kwargs + "}") or {}  # type: ignore
            except ValueError as e:
                raise ValueError(f"Cannot parse {mod_kwargs}, ensure it is json compliant.") from e

        mod_kwargs = mod_kwargs or {}  # type: ignore

        if not isinstance(mod_kwargs, dict):
            raise TypeError(f"Cannot safely use {mod_kwargs!r}.")

        self._variable = matches[0][1]
        self._modification_class = mod_type
        self._modification_dict = mod_kwargs

    def _from_dict(self, specification: dict[str, Any]):
        REQUIRED_KEYS = ["modification", "source_var"]
        for key in REQUIRED_KEYS:
            if key not in specification:
                raise KeyError(f"Missing required key: {key} in specification. {specification}.")

        self._variable = specification.pop("source_var")

        if "target_var" in specification:
            self._variable += f">{specification.pop('target_var')}"

        mod_type = specification.pop("modification")
        if mod_type not in MODIFICATION_DICT:
            raise ValueError(
                f"Cannot find modification {mod_type!r}, expected one of {list(MODIFICATION_DICT.keys())}."
            )
        self._modification_class = mod_type
        self._modification_dict = dict(specification)

    def standard_out(
        self,
    ) -> tuple[str, Optional[Type[Modification]], Optional[dict[str, Any]]]:
        return (
            self._variable,
            MODIFICATION_DICT[self._modification_class] if self._modification_class else None,
            self._modification_dict,
        )

    def __repr__(self):
        if not self._modification_class:
            return self._variable
        mod_dict_str = (
            ""
            if not self._modification_dict
            else ", ".join(f"{key!s}:{val!r}" for key, val in self._modification_dict.items())
        )
        return f"!{self._modification_class}[{mod_dict_str}]:{self._variable}"


def _get_modifications(
    variables: Union[
        str,
        dict[str, Any],
        tuple[Union[str, dict[str, Any]], ...],
        list[Union[str, dict[str, Any]]],
    ],
) -> tuple[dict[str, tuple[Type["Modification"], dict[str, Any]]], list[str]]:
    """
    Parse `variables` into a set of modification specifications, and remove mod spec

    Variables:
        Can be str of form
            ```
            - '!accumulate[period: "6 hourly"]:tcwv>accum_tcwv'
            ```
            Where the `!ABC` references the function to apply, the `[]` the init kwargs needed in json form,
            and all after `:` being the normal variable specification with anything after `>` being the new name.

        Or dictionary with following keys
        ```
            - source_var (REQUIRED)     Variable to modify
            - modification (REQUIRED)   Modification to apply
            - target_var                Rename of variable
            - **                        Any other keys for `modification`
        ```

    Raises:
        ValueError:
            If cannot find modification as specified
        TypeError:
            If parsing of modification init_kwargs cannot be safely used.

    Returns:
        (tuple[dict[str, tuple[Type['Modification'], dict[str, Any]]], list[str]]):
            Modification dictionary, list of variables stripped of modifications markers
    """
    variables = list(variables) if isinstance(variables, (tuple, list)) else [variables]

    variables_stripped: list[str] = []
    modifications: dict[str, tuple[Type[Modification], dict[str, Any]]] = {}

    for var in variables:
        specification = VariableModification(var)
        var_name, mod_type, mod_kwargs = specification.standard_out()

        if mod_type is None or mod_kwargs is None:
            variables_stripped.append(var_name)
            continue

        modifications[var_name] = (mod_type, mod_kwargs)
        variables_stripped.append(var_name.split(">")[0])

    return modifications, variables_stripped


def variable_modifications(
    variable_keyword: str = "variable",
    *,
    remove_variables: bool = False,
    skip_if_invalid_class: bool = False,
):
    """
    Allow modifications of variables dynamically,

    Args:
        variable_keyword (str, optional):
            Parameter name of variables to parse. Defaults to "variable".
        remove_variables (bool, optional):
            Whether to remove variables from the initialisation of the underlying class. Defaults to False.
        skip_if_invalid_class (bool, optional):
            Whether to skip if discovered class is invalid. Is invalid
            if class is not a subclass of TimeIndex and DataIndex

    Raises:
        KeyError:
            If cannot find `variable_keyword` in init args.
        TypeError:
            If class is not a subclass of TimeIndex and DataIndex and not `skip_if_invalid_class`.


    Syntax:
        Within the specification of the variables, a user can set the modifications with either,

        Can be str of form
            ```
            - '!accumulate[period: "6 hourly"]:tcwv>accum_tcwv'
            ```
            Where the `!ABC` references the function to apply, the `[]` the init kwargs needed in json form,
            and all after `:` being the normal variable specification with anything after `>` being the new name.

        Or dictionary with following keys
        ```
            - source_var (REQUIRED)     Variable to modify
            - modification (REQUIRED)   Modification to apply
            - target_var                Rename of variable
            - **                        Any other keys for `modification`
        ```

        This will be transparent to the user, and only act upon retrieval of data.

    Supported:
        Available modifications include:
        ```
            - !accumulate
            - !mean
            - !aggregate
        ```

    Examples:
        ```python
        class Archive(ArchiveIndex):
            @variable_modifications(variable_keyword = 'variable')
            def __init__(self, variable):
                ...

        # Then usage of that Archive
        Archive('!accumulate[period = "6 hourly"]:tcwv)
        ```

    Notes:
        If using this decorator with `check_arguments` put this one above it, and with `alias_arguments` put it below.
    """

    def internal(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get variables to parse modifications from
            all_params = _default_params(func)
            all_params.update(**kwargs, **_args_to_kwargs(func, args))

            if variable_keyword not in all_params:
                if skip_if_invalid_class:
                    return func(*args, **kwargs)
                raise KeyError(
                    f"Cannot find variables to parse. Looking for {variable_keyword!r}. Found: {list(all_params.keys())}."
                )
            variables = all_params[variable_keyword]

            if variables is None:
                return func(*args, **kwargs)

            # Get modifications
            modifications, stripped_variables = _get_modifications(variables)
            if len(modifications.keys()) == 0:
                return func(*args, **kwargs)

            if remove_variables:  # Remove variables being modified from base index
                stripped_variables = list(set(stripped_variables).difference(list(modifications.keys())))

            # Update variables to remove modifications syntax
            args, kwargs = _update_variables(func, args, kwargs, variable_keyword, stripped_variables)
            result = func(*args, **kwargs)

            index_initialised: TimeDataIndex = args[0]
            if not isinstance(index_initialised, TimeDataIndex):
                raise TypeError(
                    f"Cannot add modification to {index_initialised.__class__!r}, must be a subclass of `TimeDataIndex`."
                )

            if modifications:  # Add modifications if any detected
                index_initialised.update_initialisation(**{variable_keyword: variables})  # type: ignore

                # Add as transform
                index_initialised.base_transforms += Modifier(
                    index_initialised,
                    modifications,
                    index_kwargs=kwargs,
                    variable_keyword=variable_keyword,
                )

            return result

        return wrapper

    return internal


class Modifier(Transform):
    """
    `Transform` to apply the modification to variables
    """

    _pyearthtools_repr = {"ignore": ["index", "index_kwargs", "variable_keyword"]}

    def __init__(
        self,
        index: "TimeDataIndex",
        modifications: dict[str, tuple[Type["Modification"], dict[str, Any]]],
        index_kwargs: dict[str, Any],
        variable_keyword: str,
    ):
        """
        Setup Modifier

        Args:
            index (TimeDataIndex):
                Base Index in which data is being modified
            modifications (dict[str, tuple[Type['Modification'], dict[str, Any]]]):
                Dictionary of modifications:
                    variable: (Modification Class, modification init kwargs)
            index_kwargs (dict[str, Any]):
                Kwargs used to initialise `index`, used to recreate the indexes.
            variable_keyword (str):
                Keyword name for variable
        """
        super().__init__()
        self.record_initialisation(ignore="index")

        self._index = index
        self._index_kwargs = index_kwargs
        self._variable_keyword = variable_keyword
        self._modifications = modifications

        self._variables_being_modified = map(lambda x: x.split(">")[0], list(modifications.keys()))

    def _likely_variable_name(self, variable: str, dataset: xr.Dataset) -> str:
        """Find likely name for `variable` by looking for `Rename` Transforms"""
        variable_look = variable
        for transform in self._index.base_transforms:
            if transform.__class__.__name__ == "Rename":
                variable_look = transform._names.get(variable_look, variable_look)  # type: ignore

            if variable_look in dataset:
                return variable_look

        variable_look = variable.split("/")[-1]
        if variable_look in dataset:
            return variable_look

        return variable

    @functools.cached_property
    def modifiers(self) -> dict[str, Modification]:
        """
        Get initilised dictionary of modifiers

        variable: Modifier
        """
        return {
            var: mod(
                variable=var.split(">")[0],
                index_class=self._index,
                index_kwargs=self._index_kwargs,
                variable_keyword=self._variable_keyword,
                **kwargs,
            )
            for var, (mod, kwargs) in self._modifications.items()
        }

    def apply(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Apply modifications to `dataset`

        Will replace each variable being modified if in `dataset`.
        """
        # dataset = dataset[set(dataset.dims).difference(map(functools.partial(self._likely_variable_name, dataset = dataset), self._variables_being_modified))]

        for var, modifier in self.modifiers.items():
            base_name, new_name = var.split(">") if ">" in var else (var, None)
            base_name = self._likely_variable_name(base_name, dataset)
            dataset = dataset.drop_vars(base_name, errors="ignore")

            mod_result = modifier(dataset, base_name)
            if not dataset.coords.equals(mod_result.coords) and len(dataset.data_vars) == 0:  # type: ignore
                dataset = mod_result.to_dataset(name=new_name or base_name)
            else:
                if isinstance(mod_result, xr.Dataset):
                    mod_result = mod_result.rename({base_name: new_name or base_name})
                    for var in mod_result:
                        dataset[var] = mod_result[var]
                else:
                    dataset[new_name or base_name] = mod_result
        return dataset

    def to_repr_dict(self):
        return {"modifications": tuple(self.modifiers.values())}
