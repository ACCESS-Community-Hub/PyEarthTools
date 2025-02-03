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
Index spell checking
"""

from __future__ import annotations
from typing import Any, Type

from pyearthtools.data.exceptions import InvalidIndexError
from pyearthtools.data.indexes.utilities.fileload import open_static


class VariableDefault:
    """Variable Default Class Marker

    Used to mark arguments which may only have one possibility, but might not.
    Used within the `check_arguments` decorator.
    """

    def __repr__(self):
        return self.__class__.__name__

    pass


VARIABLE_DEFAULT = Type[VariableDefault]


def check_prompt(value: str | Any, true_values: list[str] | Any, name: str = "parameter") -> Any:
    """
    Check if `value` is in `true_values`, and if not raise an error with helpful tips

    Args:
        value (str | Any):
            Incoming Value
        true_values (list[str] | Any):
            True Values
        name (str, optional):
            Name of value. Defaults to "parameter".

    Returns:
        (Any):
            `value`, or `true_values`

            If `value` is VariableDefault, and only a single `true_values` given,
            return that `true_value`.

            If 'value' == '*', return `true_values`

    Raises:
        InvalidIndexError:
            Helpful error message prompting valid
    """
    if value in true_values:
        return value

    ## If it is variable default
    if isinstance(value, VariableDefault) or value is VariableDefault:
        if len(true_values) == 1:
            return true_values[0]
        else:
            raise InvalidIndexError(f"{name!r} must be given.\n" f"Must be one of {true_values}")

    if isinstance(value, str):
        if value == "*":
            return true_values

        if not value == "":
            prompt(value, true_values, name=name)
            return value

    elif isinstance(value, (list, tuple)) and len(value) > 0:
        for v in value:
            if v not in true_values:
                prompt(v, true_values, name=name)
        return value

    prompt(value, true_values, name=name)


def prompt(variable: str | None, truth_variable: list, name: str = "Variable"):
    """
    Find closest true value to the given value, and raise an error
    suggesting these true values.
    """

    if variable == "":
        variable = None

    if variable is None and None not in truth_variable:
        truth_variable.sort()
        raise InvalidIndexError(f"{name!s}: {variable!r} is invalid.\n" f"Did you mean one of: {truth_variable}")

    closest_variables = None

    if isinstance(variable, str):
        closest_variables = find_closest_variables(variable, truth_variable)
        if closest_variables == "":
            closest_variables = None

    if closest_variables is None or (isinstance(closest_variables, list) and len(closest_variables) == 0):
        closest_variables = truth_variable
    closest_variables.sort()

    raise InvalidIndexError(f"{name!s}: {variable!r} is invalid.\n" f"Did you mean one of: {closest_variables}")


def find_closest_variables(variable: str, truth_variable: list[str]):
    truth_variable.sort()
    closest_variables = []

    if isinstance(variable, list):
        for var in variable:
            closest_variables.append(candidates(var, truth_variable))
    else:
        closest_variables = [candidates(variable, truth_variable)]
    while None in closest_variables:
        closest_variables.remove(None)

    closest_variables = [", ".join(var) for var in closest_variables] if closest_variables else truth_variable
    return closest_variables


def format(words):
    return_words = []
    if words is None:
        return None

    for word in words:
        if word is not None:
            return_words.append(str(word))
    return return_words


def candidates(word, true_words):
    "Generate possible spelling corrections for word."
    return format(
        known([word], true_words) or known(pyearthtoolss1(word), true_words) or known(pyearthtoolss2(word), true_words) or None
    )


def known(words, true_words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in true_words)


def pyearthtoolss1(word):
    "All pyearthtoolss that are one pyearthtools away from `word`."
    letters = "abcdefghijklmnopqrstuvwxyz"
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def pyearthtoolss2(word):
    "All pyearthtoolss that are two pyearthtoolss away from `word`."
    return (e2 for e1 in pyearthtoolss1(word) for e2 in pyearthtoolss1(e1))
