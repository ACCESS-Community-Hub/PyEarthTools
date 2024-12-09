# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
Additional base data types included for use in `pyearthtools`


# TODO move to pyearthtools.utils

"""

from __future__ import annotations
from _collections_abc import dict_keys

from typing import Any
from collections import OrderedDict


class Collection(tuple):
    """
    A modified tuple type object which allows attributes and methods to be accessed.

    Attributes and methods will be returned as a `Collection`, thus allowing their attributes and methods to be accessed.

    Any item in a `Collection` can be accessed by using the `[]` syntax, and can be iterated over.


    Examples:
        >>> collec = pyearthtools.data.Collection({'item_1':10}, {'item_2':42})
        >>> collec
        Collection Containing:
            {'item_1': 10}
            {'item_2': 42}
        >>> collec.keys()
        Collection Containing:
            dict_keys(['item_1'])
            dict_keys(['item_2'])
        >>> collec[0]
        {'item_1': 10}
    """

    def __new__(cls, *args: Any, **kwds):
        obj = tuple.__new__(cls)
        return obj

    def __init__(self, *args: Any):
        self._items: tuple
        self._items = args

        self.__check_html_repr()

    def __getattr__(self, key: str) -> Collection:
        return Collection(*[getattr(item, key) for item in self._items])

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        result = [item(*args, **kwds) for item in self._items]
        if len(result) == 0:
            return result[0]
        return Collection(*result)

    def __getitem__(self, idx: int | str) -> Any:  # type: ignore
        if isinstance(idx, int):
            return self._items[idx]
        return getattr(self, idx)

    def __iter__(self) -> Any:
        for i in self._items:
            yield i

    def __len__(self):
        return len(self._items)

    def __contains__(self, key: Any):
        for item in self._items:
            if key not in item:
                return False
        return True

    # def pop(self):
    #     return self._items.pop()

    # def append(self, item):
    #     self._items.append(item)

    def __repr__(self) -> str:
        return_string = "Collection Containing: \n"
        for item in self._items:
            return_string += "\t" + item.__repr__() + "\n"
        return return_string

    def __make_repr_html_(self) -> str:
        return_string = "<h1>Collection Containing: </h1><break>"
        for item in self._items:
            return_string += f"<div> {item._repr_html_()} </div>"
        return return_string

    def __check_html_repr(self) -> None:
        for item in self._items:
            if not hasattr(item, "_repr_html_"):
                return
        self._repr_html_ = self.__make_repr_html_


class LabelledCollection(Collection):
    """
    A modified unmutable dict like object which allows attributes and methods to be accessed
    of the underlying objects, while retaining the original names.
    This allows for a name to be given to a root object, and any operations or attributes from
    said object will remain linked to that name.

    Attributes and methods will be returned as a `LabelledCollection`, thus allowing their
    attributes and methods to be accessed.

    Any item in a `LabelledCollection` can be accessed by it's given name, and can be iterated over.
    """

    def __init__(self, **kwargs: Any):
        self._items = OrderedDict(kwargs)
        self.__check_html_repr()

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

    def __getattr__(self, attr: str) -> Collection:
        return LabelledCollection(**{key: getattr(item, attr) for key, item in self._items.items()})

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return LabelledCollection(**{key: item(*args, **kwds) for key, item in self._items.items()})

    def __getitem__(self, idx: int | str) -> Any:  # type: ignore
        if idx in self._items:
            return self._items[idx]
        return getattr(self, idx)

    def items(self):
        return self._items.items()

    def keys(self) -> dict_keys:
        return self._items.keys()

    def values(self) -> dict_keys:
        return self._items.values()

    def __iter__(self) -> Any:
        for i in self._items.values():
            yield i

    ## Repr
    def __repr__(self) -> str:
        return_string = "Labelled Collection Containing: \n"
        for key, item in self._items.items():
            return_string += "\t" + f"{key} : " + item.__repr__() + "\n"
        return return_string

    def __make_repr_html_(self) -> str:
        return_string = "<h1>Labelled Collection Containing: </h1><break>"
        for key, item in self._items.items():
            return_string += f"<div><h3>{key}: </h3> {item._repr_html_()} </div>"
        return return_string

    def __check_html_repr(self) -> None:
        for item in self._items.values():
            if not hasattr(item, "_repr_html_"):
                return
        self._repr_html_ = self.__make_repr_html_
