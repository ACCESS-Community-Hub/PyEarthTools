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
Utilities
"""

from __future__ import annotations
from typing import Any, Callable

import re
import os
import inspect
import readline
import glob

import entrypoints


class Colour:  # pylint: disable=R0903
    """Colour helper"""

    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


class CategorisedObjects:
    """
    Generic class to allow access into a categorised objects.

    Categories are formed from nested kwargs and dictionaries, and can be set later with `__setitem__`.
    Key's must be hashable, just like a dictionary.

    Examples:

        >>> record = CategorisedObjects('Example', category_1 = {'sub_cat': 10})
        >>> record.category_1
        >>> ─┬ category_1 ──
        >>>   └──sub_cat

    ## Parsing

        Overriding `_parse` allows custom classes to be parsed when retrieved.
        Overriding `_name` allows custom classes names to be retrieved when displaying what is available.
    """

    SPACING = " "
    _objects: dict[str, CategorisedObjects]

    def __init__(
        self,
        name: str,
        categories: dict[str, Any | CategorisedObjects] | None = None,
        *,
        _parse: Callable | None = None,
        **objects: Any | CategorisedObjects | dict[str, Any | CategorisedObjects],
    ):
        """
        Construct a Category, can itself have sub categories.

        If any `object` is a dictionary, create another `CategorisedObjects` at that entry.

        Args:
            name (str):
                Name of this category.
            categories (dict[str, Any | CategorisedObjects] | None, optional):
                Dictionary to configure categories to allow access to.
                If element is dictionary, will be configured as a sub category. Defaults to None
            _parse (Callable, None, optional):
                Init arg to override `_parse` function, to allow parsing of object upon retrieval.
                Must be a callable expecting self and one argument.
            **objects (Any | CategorisedObjects | dict[str, Any | CategorisedObjects]):
                Kwargs form of `categories`, kwarg key is top level category.
        """
        self._cat_name = name
        if categories is None:
            categories = {}
        categories.update(objects)

        if _parse is not None:
            self._parse = _parse

        for key, value in categories.items():
            if isinstance(value, dict):
                categories[key] = CategorisedObjects(name=key, _parse=self._parse, **value)
                categories[key]._name = self._name

        self._objects = categories

    def _parse(self, value: Any) -> Any:  # pylint: disable=E0202
        """Parse object when being retrieved. By default does nothing."""
        return value

    def _name(self, value: Any) -> Any:
        """Get name of object, used when displaying what is available. By default forces to str."""
        return str(value)

    @property
    def available(self) -> tuple[str, ...]:
        """Get list of available objects"""

        def dict_to_path(d, path=""):  # Convert `models` to a list with full path
            for k, v in d.items():
                if isinstance(v, (CategorisedObjects, dict)):
                    # yield self._name(k)
                    yield from dict_to_path(v, path + self._name(k) + "/")
                else:
                    yield path + self._name(k)  # self._name(v)

        return tuple(dict_to_path(self._objects))

    def __dir__(self):
        super_dir = list(super().__dir__())
        super_dir.extend(self.available)
        super_dir.extend(self._objects.keys())
        return tuple(super_dir)

    def __getattr__(self, __key: str) -> Any | CategorisedObjects:
        """
        Get object or subcategory from this `Category`.

        If `pyearthtools.zoo.BaseForecastModel` is being retrieved, it will be loaded from the `EntryPoint`

        Can retrieve object by getting attibute one layer at a time,
        or by `getattr(self, 'category/objectName'), or if last name is unique, that name alone.


        Args:
            __key (str):
                Key or subcategory to retrieve

        Raises:
            AttributeError:
                If `__key` not found

        Returns:
            (Any | CategorisedObjects):
                Object assigned or additional sub category.
        """
        try:
            return self[__key]
        except IndexError:
            pass
        raise AttributeError(
            f"Cannot find {__key!r} in {self.__class__.__name__}.\nAvailable: {tuple(self._objects.keys())}"
        )

    def __getitem__(self, __idx: str):
        """
        Get element from category

        Can be specified as full path, final name if unique or next level
        """
        if isinstance(__idx, str) and "/" in __idx:  # Retrieves model if full categorical definition given.
            obj = self
            for k in __idx.split("/"):
                obj = getattr(obj, k)
            return obj

        if (
            __idx not in self._objects
        ):  # If not available directly in `models`, check if it is available as the last element.
            element_to_index = {x.split("/")[-1]: i for i, x in enumerate(self.available)}
            last_element = [x.split("/")[-1] for x in self.available]
            count = last_element.count(__idx)

            if __idx in element_to_index:
                if (
                    count == 1
                    and __idx in element_to_index
                    and self.available[element_to_index[__idx]] in self.available
                ):
                    return self[self.available[element_to_index[__idx]]]
                raise ValueError(
                    f"{__idx!r} was found to reference {count} entries. Specify it's categorical path completely."
                )
            raise IndexError(f"Cannot find {__idx!r} in {self.__class__.__name__}.\nAvailable: {self.available}")

        return self._parse(self._objects[__idx])

    def __setitem__(self, __idx, __value):
        """
        Set element

        Can be given as full path seperated by '/'.

        Value can be dictionary, which will be expanded.
        """
        if isinstance(__value, dict):
            __value = CategorisedObjects(__idx, **__value)

        if isinstance(__idx, str) and "/" in __idx:
            obj = self._objects
            for i, idx in enumerate(__idx.split("/")):
                if idx in obj and i < (len(__idx.split("/")) - 1):
                    if not isinstance(obj[idx], (CategorisedObjects, dict)):
                        raise TypeError(
                            f"Object at {'/'.join(__idx.split('/')[:i])!r} is not a dict type, and cannot be set."
                        )
                else:
                    obj[idx] = CategorisedObjects(idx) if i < (len(__idx.split("/")) - 1) else __value
                obj = obj[idx]
        else:
            self._objects[__idx] = __value

    def update(self, __dict: dict[Any, Any] | None = None, **kwargs: Any):
        """
        Update `CategorisedObjects`

        Can be given as full path seperated by '/'.

        Value can be dictionary, which will be expanded.
        """
        if __dict is None:
            __dict = {}

        __dict.update(kwargs)

        for key, value in __dict.items():
            self[key] = value

    ### Dictionary & List Emulation
    def keys(self):
        """Category.keys() -> a generator object providing a view on Category's keys"""
        yield from self._objects.keys()

    def values(self):
        """Category.values() -> an generator object providing a view on Category's values"""
        yield from self._objects.values()

    def items(self):
        """Category.items() -> a generator object providing a view on Category's items"""
        yield from self._objects.items()

    def __contains__(self, key: str) -> bool:
        return key in self._objects or key in self.available

    def __len__(self) -> int:
        """Return the number of items in `CategorisedObjects`."""
        return len(self._objects)

    def __iter__(self):
        yield from self.keys()

    ### Convert to str
    def __str__(self) -> str:
        return_str = f"─┬{Colour.BOLD} {self._cat_name} {Colour.END}──"

        for i, (key, val) in enumerate(self.items()):
            str_obj = val
            if not isinstance(val, CategorisedObjects):
                str_obj = key
            val_str = self._name(str_obj)
            val_str = val_str.replace("\n", f"\n{self.SPACING}{'│' if i < len(self) - 1 else ' '}")
            val_str = val_str.replace("─", "", 1) if "─" in val_str else f"{val_str}"

            symbol = "├" if i < len(self) - 1 else "└"
            seperator = "──" if "─" not in val_str else "─"
            return_str += f"\n{self.SPACING}{symbol}{seperator}{val_str}"

        return return_str

    def __repr__(self):
        return str(self)


class AvailableModels(CategorisedObjects):
    """
    Get all available models as defined by `entrypoints` underneath `pyearthtools.zoo.register`.

    Categorise with these entry points by seperating layers with `_`.

    Examples:

        >>> # Entrypoints
        >>> # NESM_modelNAME
        >>> AvailableModels()
        >>> ─┬ Available Models ──
        >>>   └─┬ NESM ──
        >>>      └──modelNAME

    Can retrieve model by getting attibute one layer at a time,
    or by `getattr(self, 'NESM/modelNAME')`, or if last name is unique, that name alone.

    If `NESM/Model` exists within the AvailableModels, it can be retrieved in the following way,
    ```python
    AvailableModels.NESM.modelNAME
    AvailableModels['NESM/modelNAME']
    AvailableModels.modelNAME # Only works if `modelNAME` is unique.
    ```
    """

    def __init__(self):
        """
        Construct object containing all available models

        Raises:
            ValueError:
                If a model will get overwritten by a duplicate key.
        """

        available_models = self._find_available_models()
        super().__init__("Available Models", **available_models)

    def _find_available_models(self) -> dict[str, Any]:
        available_models = {}

        entry_points = [
            *entrypoints.get_group_all("pyearthtools.zoo.model"),
            *entrypoints.get_group_all("pyearthtools.zoo.register"),
        ]
        entry_points.sort(key=lambda x: str(getattr(x, "name", x)))

        for e in entry_points:
            current_obj = available_models

            name = str(getattr(e, "name", e))
            elements = name.split("_")

            for i, elem in enumerate(elements):
                if (i == (len(elements) - 1) and elem in current_obj) or (
                    elem in current_obj and not isinstance(current_obj[elem], dict)
                ):
                    if name == current_obj[elem].name:
                        continue
                    raise ValueError(
                        f"Seperating {name!r} into categories would cause an override of {current_obj[elem]!s}"
                        "Cannot set"
                    )

                if elem not in current_obj:
                    current_obj[elem] = {} if i < (len(elements) - 1) else e
                current_obj = current_obj[elem]
        return available_models

    def _parse(self, value: Any | entrypoints.EntryPoint):
        if isinstance(value, entrypoints.EntryPoint):
            return value.load()
        return value

    def refresh(self):
        """Refresh available models"""
        self._objects = CategorisedObjects(  # pylint: disable=W0212
            "Available Models", **self._find_available_models()
        )._objects  # pylint: disable=W0212


BOOL_MAPPING = {"true": True, "false": False}


def parse_str(item: str) -> str | int | float | bool:
    """
    Parse a str to a boolean if represents a bool
    """
    item = item.strip()

    if item.isdigit():
        return int(item)
    try:
        return float(item)
    except Exception:  # pylint: disable=W0718
        pass
    if item.lower() in BOOL_MAPPING:
        return BOOL_MAPPING[item.lower()]
    return item


def find_demlim(value: str, delim_options: list[str]):
    """
    Find which delimiter is being used out of `delim_options`

    Defaults to '-' if none found
    """
    for delim in delim_options:
        if delim in value:
            return delim
    return "-"


def delta_conversion(value: Any, unit: str = "hour") -> int | Any:
    """
    Attempt to convert a given `value` to an integer of the given `unit`.

    If cannot convert, will quietly return `value`

    Args:
        value (Any):
            Value to convert
        unit (str, optional):
            Unit to convert in to. Defaults to 'hour'.

    Returns:
        (int):
            Time delta in unit
    """
    import pandas as pd  # pylint: disable=C0415

    delta = None

    if isinstance(value, int):
        delta = pd.Timedelta(value, "hour")

    elif isinstance(value, str):
        value = value.replace("'", "").replace('"', "")
        value = value.split(find_demlim(value, ["-", "_", ",", " "]))
        value[0] = float(value[0])

        delta = pd.Timedelta(*value)

    elif hasattr(value, "pd_timedelta"):
        delta = value.pd_timedelta

    elif isinstance(value, pd.Timedelta):
        delta = value

    elif isinstance(value, (tuple, list)) and len(value) == 1:
        return delta_conversion(value[0], unit=unit)

    if delta is None:
        return value
    return delta // pd.Timedelta(1, unit)  # type: ignore


def split_name_assignment(config: str) -> tuple[str, dict[str, str | int] | None]:
    """
    Split `config` into name and assignment components.

    Assignment is given enclosed in {}, and multiple assignments can be split by ','.

    If no assignment, return it as None

    Args:
        config (str):
            Pipeline config to parse

    Raises:
        ValueError:
            If too many elements discovered

    Returns:
        (tuple[str, dict[str, str | int] | None]):
            config name, dictionary of assignments if any or None
    """
    result = re.search(r"(.*)(\{.*\})", config)
    if result is None or len(result.groups()) == 0 or len(result.groups()) == 1:
        return config, None
    if len(result.groups()) > 2 or "{" in result.group(1):
        raise ValueError(
            f"Splitting the config: {config!r} provided too many components. Keep all assignments in one group."
        )

    assignment_components: list[list[str]] | None = None
    assignment: dict[str, Any] | None = None

    attributes = str(result.group(2))

    try:
        assignment_components = [
            list(map(lambda x: x.strip(), a.split("="))) for a in attributes.strip("{").strip("}").split(",")
        ]
        assignment = dict(assignment_components)
        assignment = {key: parse_str(val) for key, val in assignment.items()}
    except (ValueError, AttributeError):
        pass

    if assignment_components is None or assignment is None:
        raise ValueError(f"Could not parse attribute assignment: {attributes}.")

    return result.group(1), assignment


def create_mapping(list1: list[str], list2: list[str]) -> dict[str, str | None]:
    """
    Creates a dictionary mapping elements from list1 to list2, ignoring text in ().

    Allows data to be associated with pipelines designed to be generic.
    If no element found in the second list, value will be None.
    -- Generated by Bard

    Args:
        list1 (list[str]):
            A list of strings.
        list2 (list[str]):
            A list of strings.

    Returns:
        (dict[str, str | None]):
            A dictionary mapping elements from list1 to list2, ignoring text in ().

    Examples:
        Given two lists ['era5', 'era5(test)'] and ['era5'], the mapping would be

    """

    mapping = {}

    # Loop through elements in list1
    for element1 in list1:
        # Remove all characters in () from element1
        base_element1 = re.sub(r"\(.*\)", "", element1)

        # Find the corresponding element in list2
        corresponding_element = None
        for element2 in list2:
            for option in [element1, base_element1]:
                if option == element2:
                    corresponding_element = element2
                    break

        # Add the mapping to the dictionary
        mapping[element1] = corresponding_element

    return mapping


def get_annotation(val: inspect.Parameter):
    """
    Get annotation from a signature value
    """
    if val.annotation is inspect._empty:  # pylint: disable=W0212
        return val.default
    if "Literal" in str(val.annotation):
        return list(val.annotation.__args__)  # pylint: disable=W0212
    return val.annotation


def get_arguments(function: Callable) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Get arguments of a function

    Args:
        function (Callable):
            Function to get arguments of

    Returns:
        (tuple[dict[str,Any], dict[str, Any]]):
            [Required arguments, Type hints], [Defaulted arguements, defaults or type hints]
    """
    arg_spec = inspect.signature(function).parameters

    required = {}
    elements = {}

    for arg in arg_spec:
        val = arg_spec[arg]
        if val.default is inspect._empty:  # pylint: disable=W0212
            required[arg] = get_annotation(val)
            continue
        elements[arg] = get_annotation(val)
    return required, elements


class TabCompleter:
    """
    A tab completer that can either complete from the filesystem or from a list.
    """

    # Partially taken from http://stackoverflow.com/questions/5637124/tab-completion-in-pythons-raw-input
    # From https://gist.github.com/iamatypeofwalrus/5637895

    def path_completer(self, text, state):
        """
        This is the tab completer for systems paths.
        Only tested on Linux systems
        """
        _ = readline.get_line_buffer().split()

        # replace ~ with the user's home dir. See https://docs.python.org/2/library/os.path.html
        if "~" in text:
            text = os.path.expanduser(text)

        # autocomplete directories with having a trailing slash
        if os.path.isdir(text):
            text += "/"

        return [x if not os.path.isdir(x) else f"{str(x)}/" for x in glob.glob(str(text) + "*")][state]

    def create_list_completer(self, ll: list | str):
        """
        This is a closure that creates a method that autocompletes from
        the given list.

        Since the autocomplete function can't be given a list to complete from
        a closure is used to create the listCompleter function with a list to complete
        from.
        """

        if isinstance(ll, str):
            ll = [ll]

        def list_completer(_, state):
            line = readline.get_line_buffer()

            if not line:
                return [c + " " for c in ll][state]
            return [c + " " for c in ll if c.startswith(line)][state]

        return list_completer
