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
Equation evaluation and dataset transform.
"""

from __future__ import annotations
from typing import Any

import xarray as xr
import numpy as np
import operator
import math
import re
import logging


from pyearthtools.data.transforms import Transform
from pyearthtools.utils.decorators import BackwardsCompatibility

LOG = logging.getLogger("pyearthtools.data")


class EquationException(Exception):
    """Equation Exception"""

    pass


def not_nan(obj_1, obj_2):
    if isinstance(obj_1, (xr.Dataset, xr.DataArray)):
        return xr.where(obj_1.isnull(), obj_2, obj_1)
    return np.where(np.isnan(obj_1), obj_2, obj_1)


SYMBOLS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "//": operator.floordiv,
    "%": operator.mod,
    # "@": np.matmul,
    "**": np.power,
    "&": np.logical_and,
    "|": np.logical_or,
    "^": np.logical_xor,
    "and": np.bitwise_and,
    "or": np.bitwise_or,
    "xor": np.bitwise_xor,
    "not_nan": not_nan,
}

SINGLE_ELEMENT_SYMBOLS = {
    "sin": np.sin,
    "sinh": np.sinh,
    "cos": np.cos,
    "cosh": np.cosh,
    "tan": np.tan,
    "tanh": np.tanh,
    "log": np.log,
    "logtwo": np.log2,
    "logten": np.log10,
    "exp": np.exp,
    "exptwo": np.exp2,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "!": np.logical_not,
    "not": np.bitwise_not,
    "ceil": np.ceil,
    "floor": np.floor,
}

CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "g": 9.80665,  # https://en.wikipedia.org/wiki/Standard_gravity
    "E_diam": 12742 * 1000,  # Earth Diameter in metres.
}

ALLOWED_SYMBOLS_PRINT = "\n".join(
    [
        f"Two element symbols (a [op] b):\n\t{list(SYMBOLS.keys())}",
        f"Single element symbols ([op](b)):\n\t{list(SINGLE_ELEMENT_SYMBOLS.keys())}",
        f"Constants ([val]):\n\t{list(CONSTANTS.keys())}",
    ]
)


def _apply_equation(eq: str, dataset: xr.Dataset | None = None) -> tuple[xr.DataArray | float, list[str]]:
    """
    Evaluate and apply an equation.
        Use `dataset` to get variables.

    Args:
        eq (str):
            Equation to solve
        dataset (xr.Dataset | None, optional):
            Dataset to get variables from. Defaults to None.

    Raises:
        EquationException:
            Various errors occuring when evaluating the equation

    Returns:
        (tuple[xr.DataArray | float, list[str]]):
            Tuple of result, and variables used if operating on a dataset
    """
    LOG.debug(f"Evaluating: {eq!r}")

    # Split Function with spaces
    # eq = re.sub(r"([A-z\d]+)([+*//-])([A-z\d]+)", r"\1 \2 \3", eq).replace('  ', ' ')
    # eq = re.sub(r"([+*//])([A-z\d]+)", r"\1 \2", eq) # Seperate Leading symbols

    eq = re.sub(r"([A-z\d]+)([+*//])", r"\1 \2", eq)
    eq = re.sub(r"([+*//])([A-z\d]+)", r"\1 \2", eq)
    eq = re.sub(r"([A-z\d]+) -([\d]+)", r"\1 - \2", eq)

    # eq = re.sub(r"(-) (\d+)", r"\1\2", eq) # Remove issue with subtract sign
    LOG.debug(f"Altered equation to: {eq!r}")

    for func_symb in SINGLE_ELEMENT_SYMBOLS.keys():  # allow func next to bracket notation
        eq = re.sub(rf"{func_symb}([^A-z ])", rf"{func_symb} \1", eq)

    components: list[str] = eq.split(" ")
    LOG.debug(f"Discovered components: {components}")

    vars_used = []
    # Convert components into dataarrays, constants, or floats
    for i, comp in enumerate(components):
        if dataset is not None and comp in (*dataset.data_vars, *dataset.dims):
            vars_used.append(comp)
            components[i] = dataset[comp]  # type: ignore

        elif isinstance(comp, str) and comp in CONSTANTS:
            components[i] = CONSTANTS[comp]  # type: ignore

        else:
            try:
                components[i] = float(comp)  # type: ignore
            except (TypeError, ValueError):
                pass

    known_symbols = [*SYMBOLS.keys(), *SINGLE_ELEMENT_SYMBOLS.keys()]

    state: xr.DataArray | float | None = None

    def __eval_next_item(item: Any, comp: list[Any]) -> tuple[Any, list[Any]]:
        """Allow evaluation of future looking items"""
        if isinstance(item, str) and item in SINGLE_ELEMENT_SYMBOLS:
            new_item, comp = __eval_next_item(comp.pop(0), comp)
            return SINGLE_ELEMENT_SYMBOLS[item](new_item), comp
        return item, comp

    def get_next_real(comp: list[Any]) -> tuple[Any, list[Any]]:
        item = comp.pop(0)
        while item is None or (isinstance(item, str) and item == ""):
            item = comp.pop(0)
        return item, comp

    while len(components) > 0:
        item, components = get_next_real(components)

        if isinstance(item, str) and item in known_symbols:
            if state is None and item in SYMBOLS:
                raise EquationException(f"Next step called for {item!r}, but no prior values seen.")

            try:  # Compute operation
                if item in SYMBOLS:
                    next_item, components = __eval_next_item(*get_next_real(components))
                    state = SYMBOLS[item](state, next_item)

                elif item in SINGLE_ELEMENT_SYMBOLS:
                    next_item, components = __eval_next_item(*get_next_real(components))
                    state = SINGLE_ELEMENT_SYMBOLS[item](next_item)

            except (IndexError, TypeError) as e:
                raise EquationException(f"A {type(e).__name__} arose evaluating {eq!r}.") from e
        else:
            if state is None and not isinstance(item, str):
                state = item  # Set initial state
            elif state is not None and not isinstance(item, str):
                raise EquationException(
                    f"The prior state is already data: {state!r}, cannot apply another data item without an operation: {item!r}."
                )
            else:
                vars = [] if dataset is None else list((*dataset.data_vars, *dataset.dims))
                raise EquationException(
                    f"Cannot parse {item!r}. Ensure spaces seperate each component.\nDiscovered Variables:\t{vars}\nValid symbols are \n{ALLOWED_SYMBOLS_PRINT}"
                )

    if state is None:
        raise EquationException(f"After evaluating {eq!r}, the result was {None!r}.")
    return state, vars_used


def _apply_equation_with_brackets(eq: str, dataset: xr.Dataset | None = None) -> tuple[xr.DataArray | float, list[str]]:
    """
    Evaluate and apply an equation that contains brackets.
        Use `dataset` to get variables.

    Seperates each bracket, taking the lowest first, and sequentially solves.
    If operating on a Dataset, replace component with a temporary variable

    Args:
        eq (str):
            Equation to solve
        dataset (xr.Dataset | None, optional):
            Dataset to get variables from. Defaults to None.

    Raises:
        EquationException:
            If an error occurs when evaluating the equations

    Returns:
        (tuple[xr.DataArray | float, list[str]]):
            Tuple of result, and variables used if operating on a dataset
    """

    REGEX_EXPRESSION = r"\(([^\(]*?)\)"  # Will match lowest bracket
    temp_var_index = 0

    dataset = xr.Dataset(dataset) if dataset else dataset
    drop_vars = []
    temp_vars = []

    while "(" and ")" in eq:
        for match in re.finditer(REGEX_EXPRESSION, eq):
            # Get sub equation
            group = match.group()
            sub_eq = group.replace("(", "").replace(")", "")

            # Assign result to temp variable
            var_name = f"temp_var_{temp_var_index}"
            temp_vars.append(var_name)
            temp_var_index += 1

            LOG.debug(f"Setting {var_name} to result of {sub_eq!r}")

            try:
                result, eq_vars = _apply_equation(sub_eq, dataset)
                if isinstance(result, xr.DataArray) and dataset is not None:
                    dataset[var_name] = result
                    eq = eq.replace(group, f" {var_name}")
                else:
                    eq = eq.replace(group, f" {str(result).replace(' ','')}")

            except EquationException as e:
                raise EquationException(f"When processing sub equation {group!r} of {eq!r}, an error occured.") from e
            list(drop_vars.append(var) for var in eq_vars)

    result, eq_drop_vars = _apply_equation(eq, dataset)
    list(drop_vars.append(var) for var in eq_drop_vars)

    return result, drop_vars


def _evaluate(eq: str, *, dataset: xr.Dataset | None = None) -> tuple[xr.DataArray | float, list[str]]:
    """
    Evaluate a given equation
        Use `dataset` to set variables.

    Each numerical or reference component in an equation must be seperated by a space.

    If using function based symbols like 'sqrt' or 'sin', the next item will be evaluated using said function.
    These functions can be given with brackets next to them.

    Without brackets given, the equation will be evaluated left -> right.

    E.g.
        ```
        evaluate('9 * 5')
        # (45.0, [])
        evaluate('cos(pi)')
        # (-1.0, [])
        ```

    Args:
        eq (str):
            Equation to solve
        dataset (xr.Dataset | None, optional):
            Dataset to get variables from. Defaults to None

    Raises:
        EquationException:
            If a mismatch between count of '(' and ')'.
        EquationException:
            Any errors or issues parsing an equation.

    Returns:
        (tuple[xr.DataArray | float, list[str]]):
            Tuple of result, and variables used if operating on a dataset.


    Examples:
        >>> evaluate('9 * 2')
        # (18.0, [])
        >>> evaluate('sqrt(4)')
        # (2.0, [])
        >>> evaluate('sqrt(4) + 1')
        # (3.0, [])

        # Operate on datasets
        >>> evaluate('9 * (2 * variable)', dataset = ds)
        # (Result of: 9 * (2 * variable), ['variable'])
        >>> evaluate('sqrt(variable)', dataset = ds)
        # (Square root of variable, ['variable'])
    """
    if not eq.count("(") == eq.count(")"):
        raise EquationException(f"Equation brackets do not close: {eq!r}.")

    if "(" and ")" in eq:
        return _apply_equation_with_brackets(eq, dataset=dataset)
    return _apply_equation(eq, dataset=dataset)


def evaluate(eq: str, *, dataset: xr.Dataset | None = None) -> xr.DataArray | float:
    """
    Evaluate a given equation
        Use `dataset` to set variables.

    Each numerical or reference component in an equation must be seperated by a space.

    If using function based symbols like 'sqrt' or 'sin', the next item will be evaluated using said function.
    These functions can be given with brackets next to them.

    Without brackets given, the equation will be evaluated left -> right.

    Args:
        eq (str):
            Equation to solve
        dataset (xr.Dataset | None, optional):
            Dataset to get variables from. Defaults to None

    Returns:
        (xr.DataArray | float):
            Result of equation
    """
    return _evaluate(eq, dataset=dataset)[0]


def derive_equations(
    dataset: xr.Dataset,
    equation: dict[str, str | tuple[str, dict[str, Any]]] | None = None,
    *,
    drop: bool = False,
    **equations,
) -> xr.Dataset:
    """
    Derive new variables from specified `equation`/s, and set variables in the `dataset` accordingly

    Args:
        dataset (xr.Dataset):
            Dataset to get variables from, and to set new ones on
        equation (dict[str, str | tuple[str, dict[str, Any]]] | None, optional):
            Dictionary of equations, key represents new variable name.
            Can be tuple to set equation, and attribute update dictionary.
            Defaults to {}.
        drop (bool, optional):
            Drop variables used in calculations. Defaults to False.

    Returns:
        (xr.Dataset):
            Dataset with equations applied to it
    """
    if equation is None:
        equation = {}
    equation.update(dict(equations))

    drop_vars = []

    for key, eq in equation.items():
        attrs = {"long_name": f"Result of: {eq if not isinstance(eq, (list, tuple)) else eq[0]}"}

        if isinstance(eq, (list, tuple)):
            eq, _up_attrs = eq
            attrs.update(_up_attrs)
        attrs["equation"] = eq

        LOG.debug(f"Setting {key!r} to result of {eq!r}.")
        result, eq_drop_vars = _evaluate(eq, dataset=dataset)

        if key in list(dataset.coords.keys()):
            dataset = dataset.assign_coords({key: result})
        else:
            dataset[key] = result

        if attrs.pop("drop", drop):
            _ = list(drop_vars.append(var) for var in eq_drop_vars)

        dataset[key].attrs.update(**attrs)

    # Drop variables used in the calculation
    dataset = dataset.drop(set(drop_vars).intersection(dataset.data_vars), errors="ignore")  # type: ignore

    return dataset


class Derive(Transform):
    """Derive new variables"""

    def __init__(
        self,
        equation: dict[str, str | tuple[str, dict[str, Any]]] | None = None,
        drop: bool = False,
        **equations: str | tuple[str, dict[str, Any]],
    ):
        """
        Derive new variables from a dataset using provided equations.

        Allows other variables to be used by indicating only their name, and than evaluated accordingly.

        Each numerical or reference component in an equation must be seperated by a space.

        If using function based symbols like 'sqrt' or 'sin', the next item will be evaluated using said function.
        These functions can be given with brackets next to them.

        Without brackets given, the equation will be evaluated left -> right.

        E.g.
            ```
            equation = {'new_variable' : 'old_variable_1 * old_variable_2'}
            equation = {'new_variable' : 'sqrt(old_variable_1 * old_variable_2)'}
            ```

        !!! Warn
            This will evaluate an equation left -> right, but respects brackets.

            'var_1 + 9.8 * var_2' != 'var_1 + (9.8 * var_2)'

        !!! Warning
            Components of an equation should be split by ' ', a whitespace.
            ```
            a*b   # Bad
            a * b # Good
            ```

        Args:
            equation (dict[str, str  |  tuple[str, dict[str, str]]] | None, optional):
                Equation configuration. If str, equation is evaluated.
                If tuple, first element is assumed to be equation, and the second a
                dictionary to update the new vars attributes with. Defaults to None.
            drop (bool, optional):
                Drop variables used in the calculation. Can be overwritten per equation, by
                setting `drop` in attributes dictionary. Defaults to False.
            **equations (dict[str, str  |  tuple[str, dict[str, str]]], optional):
                Keyword arg form of `equation`.

        Raises:
            EquationException:
                If equation cannot be parsed

        Returns:
            (Transform):
                Transform to apply derivation.

        Examples:
            >>> derive(new_variable = 'old_variable_1 * old_variable_2', drop = True)
            # Create a `new_variable` as the product of the old two.
            >>> derive(new_variable = 'old_variable_1 * 9.8', drop = True)
            # Scale `old_variable_1` by 9.8
            >>> derive(new_variable = ('old_variable_1 * 9.8', {'long_name': 'Scaled old_variable_1'}, drop = False)
            # Scale `old_variable_1` by 9.8, and update the `long_name` to be 'Scaled old_variable_1', leaving the old var there
            >>> derive(new_variable = 'old_variable_1 - old_variable_2 * 9.8', drop = True)
            # Set `new_variable` as the difference scaled by 9.8. In effect acts as (old_variable_1 - old_variable_2) * 9.8
            >>> derive(new_variable = 'old_variable_1 - (9.8 * old_variable_2)', drop = True)
            # Multiply `old_variable_2` by 9.8 and than find difference with `old_variable_1`
            >>> derive(new_var == 'sqrt(old_var)')
            # Square root of the old variable
        """
        super().__init__()
        self.record_initialisation()

        if equation is None:
            equation = {}
        equation.update(dict(equations))

        if not isinstance(equation, dict):
            raise TypeError(
                f"`equation` must be a dictionary specifying new variable name and equation, not {type(equation)}."
            )

        equation = equation.copy()
        equation.update(dict(equations))
        if len(equation.keys()) == 0:
            raise ValueError("Must provide at least one equation.")

        self._equation = equation
        self._drop = drop

    # @property
    # def _info_(self):
    #     return dict(**self._equation, drop=self._drop)

    def apply(self, dataset: xr.Dataset) -> xr.Dataset:
        return derive_equations(dataset, drop=self._drop, equation=self._equation)


@BackwardsCompatibility(Derive)
def derive(*args, **kwargs) -> Transform: ...


# def equations(equation: str):
#     class KnownEquations(Transform):
#         @property
#         def _info_(self) -> Any | dict:
#             return dict(equation = equation)
#         def apply(self, dataset: xr.Dataset) -> xr.Dataset:
#             return super().apply(dataset)
