# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

"""
Index spell checking
"""

from __future__ import annotations
from typing import Any, Type

from edit.data.exceptions import InvalidIndexError
from edit.data.indexes.utilities.fileload import open_static


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

    elif isinstance(value, (list, tuple)):
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
    
    if variable == '':
        variable = None
    if variable is None:
        raise InvalidIndexError(f"{name!s}: {variable!r} is invalid.\n" f"Did you mean one of: {truth_variable}")

    closest_variables = find_closest_variables(variable, truth_variable)
    if closest_variables == "":
        closest_variables = truth_variable

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
        known([word], true_words) or known(edits1(word), true_words) or known(edits2(word), true_words) or None
    )


def known(words, true_words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in true_words)


def edits1(word):
    "All edits that are one edit away from `word`."
    letters = "abcdefghijklmnopqrstuvwxyz"
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))
