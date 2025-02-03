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


import copy
from typing import Literal, Sequence, Any, Optional, TypeVar


import pyearthtools.utils
from pyearthtools.utils.initialisation import init_parsing

Self = TypeVar("Self", Any, Any)

pyearthtools_INIT_KEYS = Literal["class"]
pyearthtools_REPR_KEYS = Literal["ignore", "expand", "expand_attr"]


class ReprInformation:
    def __init__(self, name, info, doc=""):
        self.__doc__ = doc
        self._name_ = name
        self._info_ = info


def parse_repr_object(obj: Any, name: str, doc: Optional[str] = None) -> ReprInformation:
    if isinstance(obj, dict):

        def clean_dict(dictionary):
            dictionary = dict(dictionary)
            for key, val in dictionary.items():
                if isinstance(val, InitialisationRecordingMixin):
                    dictionary[key] = {type(val).__name__: clean_dict(val.to_repr_dict())}
            return dictionary

        return ReprInformation(name, clean_dict(obj), doc=doc or "")

    elif isinstance(obj, InitialisationRecordingMixin):
        return parse_repr_object(obj.to_repr_dict(), name)

    elif isinstance(obj, (list, tuple)):
        new_dict = {}
        for o in obj:
            i = 0

            last_module = str(o.__module__).replace(f"{type(o).__name__}", "").split(".")[-1]
            obj_name = f"{last_module}.{type(o).__name__}".removeprefix(".")
            new_name = str(obj_name) + "[{i}]"

            while obj_name in new_dict:
                i += 1
                obj_name = str(new_name).format(i=i)
            new_dict[obj_name] = o

        return parse_repr_object(new_dict, name=name, doc=doc)

    if not isinstance(obj, dict):
        obj = {"value": obj}
    return ReprInformation(name, obj)


class InitialisationRecordingMixin:
    """Mixin to record initialisation arguments of the child class.

    Also provides a `repr` from these initialisation args.

    Children must call the following for the functionality to work.
    ```python
    super().__init__()
    self.record_initialisation()
    ```

    Properties:
        _pyearthtools_initialisation (dict[pyearthtools_INIT_KEYS, Any]):
            'class' (str): Override for class location.
        _pyearthtools_repr (dict[pyearthtools_REPR_KEYS, Any]):
            ignore (Sequence[str]): Arguments to ignore from `_initialisation`.
            expand (Sequence[str]): Arguments to expand from `_initialisation`.
            expand_attr (Sequence[str]): Arguments to get from class and expand.
        _desc_ (dict[str, Any]):
            Description of object to display at the top of the repr.
            Use 'singleline' to change element not minimised next to `Description`.
        _property (str):
            Property to call on `self` to get back correct object when loaded in from yaml.

    Methods:
        `to_repr_dict`:
            How to display object. Must return a dictionary.
    """

    _initialisation: Optional[dict[str, Any]] = None
    _pyearthtools_initialisation: dict[pyearthtools_INIT_KEYS, Any]
    _pyearthtools_repr: dict[pyearthtools_REPR_KEYS, Any]

    _desc_: Optional[dict[str, Any]]
    _property: Optional[str] = None

    @property
    def initialisation(self) -> dict[str, Any]:
        return self._initialisation or {}

    @initialisation.setter
    def initialisation(self, val: dict[str, Any]):
        self._initialisation = val

    def update_initialisation(self, update: Optional[dict[str, Any]] = None, /, **upda: Any):
        """
        Update components of the initialisation dictionary

        Args:
            update (Optional[dict[str, Any]], optional):
                Dictionary to update with. Defaults to None.
            **upda (Any):
                Kwarg form of `update`.
        """
        if self._initialisation is None:
            self._initialisation = {}

        update = update or {}
        update.update(upda)
        if "args" in update:
            update["__args"] = update.pop("args")
        self._initialisation.update(update)

    def record_initialisation(self, ignore: Optional[Sequence[str]] = None):
        """
        Record initialisation of class

        `super().__init__()` must be called before.

        Args:
            ignore (Sequence[str], optional):
                Ignore arguments. Defaults to [].
        """
        self.initialisation = init_parsing.get_initialise_args(self, ignore=ignore or [])

    def copy(self: Self, **overrides: Any) -> Self:
        """Using recorded initialisation create a copy of `self`."""
        init_kwargs = dict(self.initialisation)
        init_kwargs.update(overrides)
        return type(self)(*init_kwargs.pop("__args", ()), **init_kwargs)

    def to_repr_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary ready for repr
        """
        config = getattr(self, "_pyearthtools_repr", {})

        init_kwargs = dict(self.initialisation)
        if "ignore" in config:
            for i in config["ignore"]:
                init_kwargs.pop(i, None)
        return init_kwargs

    def __get_items_for_repr(self):

        init_kwargs = copy.copy(self.initialisation)
        if "__args" in init_kwargs:
            init_kwargs["args"] = init_kwargs.pop("__args")

        config = getattr(self, "_pyearthtools_repr", {})

        expanded_information = []

        if "expand" in config:
            for ex in (
                ex
                for ex in config["expand"]
                if str(ex).split("@", maxsplit=1)[-1] in init_kwargs and init_kwargs[str(ex).split("@", maxsplit=1)[-1]]
            ):
                name, key = str(ex).split("@", maxsplit=1) if "@" in str(ex) else (ex, ex)
                expanded_information.append(parse_repr_object(init_kwargs.pop(key), str(name)))

        if "ignore" in config:
            for i in config["ignore"]:
                init_kwargs.pop(i, None)

        if "expand_attr" in config:
            for ex in (ex for ex in config["expand_attr"] if hasattr(self, str(ex).split("@", maxsplit=1)[-1])):
                name, key = str(ex).split("@", maxsplit=1) if "@" in str(ex) else (ex, ex)
                if getattr(self, key) is not None:
                    expanded_information.append(parse_repr_object(getattr(self, key), str(name)))

        if not hasattr(self, "_desc_"):
            doc = (getattr(self, "__doc__", "") or "").strip().split("\n")[0]
        else:
            doc = ""
        return parse_repr_object(init_kwargs, "Initialisation", doc=doc), *expanded_information

    def __repr__(self):
        return pyearthtools.utils.repr_utils.standard(
            *self.__get_items_for_repr(),
            name=self.__class__.__qualname__,
            description=dict(getattr(self, "_desc_", {})),
            documentation_attr="__doc__",
            info_attr="_info_",
            name_attr="_name_",
        )

    def _repr_html_(self):
        try:
            steps = self.__get_items_for_repr()
        except NotImplementedError:
            return repr(self)

        backup_repr = self.__repr__() or "HTML Repr Failed"

        return pyearthtools.utils.repr_utils.html(
            *steps,
            name=self.__class__.__qualname__,
            description=dict(getattr(self, "_desc_", {})),
            documentation_attr="__doc__",
            name_attr="_name_",
            info_attr="_info_",
            backup_repr=backup_repr,
            expanded=True,
        )


__all__ = ["InitialisationRecordingMixin"]
