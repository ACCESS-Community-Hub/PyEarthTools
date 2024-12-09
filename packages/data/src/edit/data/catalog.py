# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
`pyearthtools` Catalog's

Record information about an [index][pyearthtools.data.indexes] or other class, and allow it to be saved and loaded to/from disk.

This [CatalogEntry][pyearthtools.data.CatalogEntry] can be called like any other index, with [Catalog][pyearthtools.data.Catalog]
automatically returning a [Collection][pyearthtools.data.Collection] of data if multiple entries recorded.

If an [index][pyearthtools.data.indexes] has called `record_initialisation`, a [Catalog][pyearthtools.data.CatalogEntry] is accessible at `.catalog`,
and the [catalog][pyearthtools.data.CatalogEntry] is used as part of the repr.

Any class can specify the property `to_init_dict` to work with the Catalog. This function must return a single element dictionary,
with the key being the class, and the value of the following form,
```python
{
    CLASS:
        { # All are optional
        args: #Arguments to initalise with
        kwargs: #Keyword arguments to initalise with
        name: #Name of entry
        }

}
```
"""

from __future__ import annotations
from collections import OrderedDict

import yaml
import json
from pathlib import Path
from typing import Any, Callable, Optional
import types
import warnings
from functools import lru_cache

from pyearthtools.utils.parsing import function_name
from pyearthtools.utils.imports import dynamic_import

import pyearthtools.data
from pyearthtools.data.collection import Collection, LabelledCollection

from pyearthtools.utils.decorators import alias_arguments

UTILS_REPR = False
try:
    import pyearthtools.utils

    UTILS_REPR = True
except ImportError:
    UTILS_REPR = False

FILE_EXTENSION = ".catalog"


def get_name(obj: Any) -> str:
    """Get name of object"""
    if isinstance(obj, str):
        return obj
    elif isinstance(obj, types.FunctionType):
        return obj.__name__
    elif isinstance(obj, type):
        return obj.__class__.__name__
    elif hasattr(obj, __name__):
        return str(obj.__name__)
    return str(obj)


class CatalogEntry:
    """
    Catalog Entry
    """

    item_class: Optional[Callable] = None

    @alias_arguments(item_class="data_index")  # Backwards compatibility
    def __init__(
        self,
        item_class: Callable,
        args: list[Any] = [],
        *extra_args,
        name: str | None = None,
        class_path: str | None = None,
        kwargs: dict = {},
        **extra_kwargs,
    ) -> None:
        """
        Setup Catalog Entry.

        Can be used to catalog any class, and the args and kwargs to initalise it.

        Args:
            item_class (Callable | str):
                Class to setup catalog entry for
            *args (Any):
                args to be passed to `item_class`
            name (str):
                Name of this entry
            class_path (str, None):
                Override for class path. If not given will be auto found.
            **kwargs (Any):
                kwargs to be passed to `item_class`
        """
        if isinstance(item_class, str):
            item_class = dynamic_import(item_class)

        for i, arg in enumerate(args):
            if isinstance(arg, str) and arg == "None":
                args[i] = None

        for key, val in kwargs.items():
            if isinstance(val, str) and val == "None":
                kwargs[key] = None

        self.name = name
        self.item_class = item_class
        self._class_path = class_path

        args = [*list(args), *list(extra_args)]

        kwargs = OrderedDict(**kwargs)

        kwargs.update(extra_kwargs)
        self._args = args
        self._kwargs = kwargs

    @property
    @lru_cache(None)
    def function(self) -> Any:
        """
        Get underlying Class

        Returns:
            (Any): Underlying Class
        """
        if self.item_class is None:
            raise AttributeError("The item_class has not been provided, cannot get `function`.")
        return self.item_class(*self._args, **self._kwargs)

    def __getattr__(self, attribute: str):
        if attribute in ["item_class", "catalog", "function"]:
            raise AttributeError(f"Catalog has no attribute: {attribute!r}")

        item_class = self.function
        if not hasattr(item_class, attribute):
            raise AttributeError(f"{item_class.__class__} has no attribute: {attribute!r}")

        # def helper_function(*args, **kwargs):
        #     return getattr(self.data_index, attribute)(
        #         *args, **kwargs  # , **self.prefilled_kwargs
        #     )

        return getattr(item_class, attribute)

    @property
    def _item_module_path(self):
        return self._class_path or function_name(self.item_class)

    def to_dict(self) -> dict:
        """
        Convert `CatalogEntry` into dict

        Returns:
            dict: Dictionary containing all info needed to reconstruct
            Structure:
                item_class: Function class path
                name: Catalog Entry name
                args: Args used to init
                kwargs: Kwargs used to init
        """
        d = {}
        d["item_class"] = self._item_module_path

        d["name"] = self.name
        d["args"] = [arg for arg in self._args]

        def convert(**kwargs):
            """Prepare kwargs

            Converts non serialisable to serialisable
            """
            for key, value in kwargs.items():
                if isinstance(value, (pyearthtools.data.TimeDelta, pyearthtools.data.TimeResolution, Path)):
                    kwargs[key] = str(value)
            return kwargs

        d["kwargs"] = convert(**{key: value for key, value in self._kwargs.items()})
        return d

    @staticmethod
    def from_dict(init_dict: dict, **kwargs) -> "CatalogEntry":
        """Create `CatalogEntry` from dictionary

        This dictionary can be of two forms, one that is the result of `CatalogEntry.to_dict()`,
        and the other a more general form.

        ```python
        ## Form of the init_dict

        {
            CLASS:
                { # All are optional
                args: #Arguments to initalise with
                kwargs: #Keyword arguments to initalise with
                name: #Name of entry
                }

        }
        ```

        Args:
            init_dict (dict):
                Initialisation Dictionary.
            **kwargs (dict, optional):
                Kwargs to replace init_dict['kwargs'] with. Defaults to {}


        Returns:
            CatalogEntry:
                Loaded `CatalogEntry`
        """

        if "data_index" in init_dict:
            init_dict["item_class"] = init_dict.pop("data_index")

        if "item_class" in init_dict:
            init_kwargs = init_dict.get("kwargs", {})
            init_kwargs.update(kwargs)

            init_class = init_dict["item_class"]

            return CatalogEntry(
                init_class,
                *init_dict.get("args", []),
                name=init_kwargs.pop("name", get_name(init_class)),
                **init_kwargs,
            )

        elif len(init_dict.keys()) == 1:
            for key, value in init_dict.items():
                value.update(kwargs)

                if "item_class" in value or "data_index" in value:
                    return CatalogEntry.from_dict(value)

                name = value.pop("name", get_name(key))

                if isinstance(key, str):
                    key = dynamic_import(key)
                return CatalogEntry(key, **value, name=name)

        raise ValueError(
            f"Unable to parse {init_dict}, either give `args`, `kwargs` form, of key of class and value of `kwargs`."
        )

    def save(self, output_file: str | Path | None = None, direct_load: bool = True) -> None | dict:
        """
        Save this `CatalogEntry` as a catalog at Path

        Args:
            output_file (str | Path | None, optional):
                Path to savefile. Defaults to None.
            direct_load (bool, optional):
                When loading this catalog entry, should the index be directly returned
                Defaults to True.
        """

        saving_catalog = Catalog(self)
        return saving_catalog.save(output_file, direct_load=direct_load)

    def set_kwargs(self, **kwargs):
        """
        Add extra kwargs

        Args:
            **kwargs (Any): Extra kwargs
        """
        for arg, value in kwargs.items():
            self._kwargs[arg] = value

    def del_kwargs(self, key: str):
        """
        Remove kwargs

        Args:
            key (str): Key to remove

        Raises:
            KeyError: If key not found
        """
        if key not in self._kwargs:
            raise KeyError(f"'{key}' not in prefilled_kwargs for entry {self.name}")
        self._kwargs.pop(key)

    def __call__(self, *args, **kwargs) -> Any:
        """
        Call underlying `function`

        Returns:
            Any: Result of `function`
        """
        return self.function(*args, **kwargs)

    def __getitem__(self, *args, **kwargs) -> Any:
        """
        Get from underlying object.

        Returns:
            Any: Result of `object.__getitem__`
        """
        return self.function.__getitem__(*args, **kwargs)

    def __str__(self) -> str:
        return f"Catalog entry for {self._item_module_path}"

    @property
    def _doc_(self):
        return self.item_class

    @property
    def _name_(self):
        return f"{self.name} - {self._item_module_path}"

    @property
    def _info_(self):
        args = {"args": self._args}
        args.update(self._kwargs)
        return args

    def __add__(self, other):
        return Catalog(self, other)

    def __repr__(self) -> str:
        if not UTILS_REPR:
            return str(self)

        return pyearthtools.utils.repr_utils.standard(
            self,
            name="Catalog Entry",
            documentation_attr="_doc_",
            name_attr="_name_",
            info_attr="_info_",
        )

    def _repr_html_(self) -> str:
        if not UTILS_REPR:
            return repr(self)

        return pyearthtools.utils.repr_utils.html(
            self,
            name="Catalog Entry",
            documentation_attr="_doc_",
            name_attr="_name_",
            info_attr="_info_",
        )


class Catalog:
    """Keep a Catalog of Data Sources

    Used to track known kwargs for functions

    Can be used for any class with specifies the function `to_init_dict`, which
    returns a dictionary with the key being the fully featured class name, and the value,
    a dictionary with the init kwargs. A `name` kwarg specifies the `CatalogEntry` name.
    """

    def __init__(self, *args, name: str | None = None, **kwargs):
        """
        Initalise a new Catalog of Data Sources

        Args:
            *args (str, Path, 'Catalog', CatalogEntry | pyearthtools.data.Index):
                Entries to add to catalog
            name (str, optional):
                Name for this catalog. Defaults to None.
            **kwargs (str, Path, 'Catalog', CatalogEntry | pyearthtools.data.Index):
                Named entries to add to catalog

        Examples:
            >>> test_catalog = Catalog()
            >>> def return_function(x, **kwargs):
            ...     return '-'.join([x,*list(kwargs.keys())])
            >>> test_catalog.append(CatalogEntry(return_function, name = 'Test' wow = 1))
            >>> test_catalog.Test('entry')
            'entry-wow'


        """
        self._catalog = OrderedDict()

        self.name = name

        for arg in args:
            self.append(arg, name=None)

        for name, entry in kwargs.items():
            self.append(entry, name=name)

    def append(
        self,
        other: "str | Path | Catalog | CatalogEntry | pyearthtools.data.DataIndex | dict",
        *,
        name: str | None = None,
    ):
        """
        Append Elements to Catalog.

        Args:
            other (str, Path, Catalog, CatalogEntry | pyearthtools.data.DataIndex | dict):
                Items to add to Catalog
            name (str, optional):
                Override for name of entry. Defaults to None.

        Raises:
            KeyError:
                If `pyearthtools.data.Index` has no attr `catalog`
            TypeError:
                If other not recognised
        """

        if isinstance(other, (Path, str)):
            opened_cat = Catalog.load(other)
            self.append(opened_cat, name=name)

        elif isinstance(other, (Catalog, list, tuple)):
            for i in other:
                self.append(i, name=name)

        elif isinstance(other, CatalogEntry):
            name = str(name or other.name)

            if name in self._catalog:  # Ensure unique name
                i = 0
                new_name = str(name) + "_{i}"
                name = str(new_name).format(i=i)
                while name in self._catalog:
                    i += 1
                    name = str(new_name).format(i=i)

            other.name = name

            self._catalog[name] = other
            setattr(self, name, other)

        elif isinstance(other, pyearthtools.data.DataIndex) or (
            hasattr(other, "catalog") and isinstance(other.catalog, (CatalogEntry, Catalog))
        ):  # Allow any class to have a `catalog` property.
            self.append(other.catalog, name=name or other.catalog.name or get_name(other))

        elif isinstance(other, pyearthtools.data.DataIndex) and not hasattr(other, "catalog"):
            raise AttributeError(
                f"Object to append appears to be a `pyearthtools.data.DataIndex`, but has no `catalog`.\n"
                "Ensure, `.record_initialisation()` has been run in the `__init__` of the Index."
            )

        elif isinstance(other, dict):
            # Load a catalog with multiple entries
            if len(other) > 1 and (
                "data_index" not in other and "item_class" not in other
            ):  # Needs a check to ensure its not accidently iterating over an entry itself
                for key, value in other.items():
                    self.append(CatalogEntry.from_dict(value, name=key))
            else:  # Single entry
                self.append(CatalogEntry.from_dict(other, name=name))

        elif hasattr(other, "to_init_dict"):  # Allow any class to specify an `to_init_dict` to make catalog of.
            self.append(other.to_init_dict(), name=name)

        else:
            raise TypeError(f"{type(other)} unable to be added. Is an invalid type and cannot find `to_init_dict`.")

    def remove(self, key: str):
        """
        Remove element from catalog

        Args:
            key (str): Key to remove

        Raises:
            KeyError: If Key not catalog
        """
        if key not in self._catalog:
            raise KeyError(f"'{key}' not in {self.name} Catalog")
        self._catalog.pop(key)
        delattr(self, key)

    def pop(self, key: str) -> CatalogEntry:
        """
        Pop element from Catalog

        Args:
            key (str): Key to pop

        Raises:
            KeyError: If key not in catalog

        Returns:
            CatalogEntry: Popped entry
        """
        if key not in self._catalog:
            raise KeyError(f"'{key}' not in {self.name} Catalog")
        delattr(self, key)
        return self._catalog.pop(key)

    def to_dict(self):
        """Get catalog as dictionary"""
        return {key: value.to_dict() for key, value in self._catalog.items()}

    def save(self, output_file: str | Path | None = None, direct_load: bool = False) -> None | dict:
        """
        Save Catalog to specified file

        Auto converts any function pointers to fully qualified path

        Args:
            output_file (str | Path | None, optional):
                Save file path. Defaults to None.
            direct_load (bool, optional):
                If Catalog contains one entry, this flag can be used so that when the catalog is loaded,
                the index is returned instead.
                Defaults to False
        """
        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # if output_file.suffix:
            #     output_file = Path(str(output_file).replace(output_file.suffix, FILE_EXTENSION))
            if not output_file.suffix:
                output_file = Path(str(output_file) + FILE_EXTENSION)

        save_catalog = self.to_dict()

        if len(self) > 1 and direct_load:
            direct_load = False

        save_catalog["direct_load"] = direct_load
        save_catalog["VERSION"] = pyearthtools.data.__version__

        if output_file:
            try:
                with open(output_file, "w") as file:
                    yaml.dump(save_catalog, file, sort_keys=False)
            except PermissionError as e:
                warnings.warn(
                    f"Could not save to {output_file!s}, due to a PermissionError",
                    UserWarning,
                )
            return
        return save_catalog

    @staticmethod
    def load(
        catalog_to_load: str | Path | dict[str, Any],
        direct_load: bool = False,
        **kwargs,
    ) -> "Catalog | Callable":
        """
        Load saved catalog file into new catalog object

        !!! Tip:
            If pointed at a folder, will search the folder looking for a catalog file of `.cat`.
            If found that catalog will be loaded, instead.
            Used to create folders loadable from pyearthtools.

        Args:
            catalog_to_load (str | Path): Filepath to catalog file
                All function pointers are converted from str to function pointer
            direct_load (bool, optional):
                If Catalog contains one entry, this flag can be used to return that index instead.
                Defaults to False

        Raises:
            FileNotFoundError: If file does not exist

        Returns:
            Catalog: Loaded Catalog
        """

        if isinstance(catalog_to_load, (str, Path)):
            catalog_to_load = Path(catalog_to_load)
            if not catalog_to_load.exists():
                raise FileNotFoundError(f"'{catalog_to_load!s}' does not exist")
            # if not input_file.suffix == FILE_EXTENSION:
            #     raise RuntimeError(f"{input_file} cannot be loaded as catalog, must be of {FILE_EXTENSION}")

            loaded_catalog = None

            if catalog_to_load.is_dir():
                possible_catalogs = list(catalog_to_load.glob("catalog.cat"))
                if len(possible_catalogs) == 1:
                    return Catalog.load(possible_catalogs[0])
                if len(possible_catalogs) > 1:
                    raise FileNotFoundError(f"Multiple catalogs found in {catalog_to_load!s}")
                raise FileNotFoundError(
                    f"Given file was actually a directory, and no valid `pyearthtools` catalog was found inside."
                    "\nEnsure the catalog ends in '.cat'"
                )

            with open(catalog_to_load) as file:
                loaded_catalog = yaml.load(file, yaml.Loader)

            if loaded_catalog is None:
                raise ValueError(f"Cannot load file: '{catalog_to_load!s}' as a catalog.")

        elif (catalog_to_load, dict):
            loaded_catalog = catalog_to_load
        else:
            raise TypeError(f"Cannot load {type(catalog_to_load)} as a Catalog.")

        direct_load = loaded_catalog.pop("direct_load", direct_load)
        version = loaded_catalog.pop("VERSION", None)

        new_catalog = Catalog()
        try:
            new_catalog.append(loaded_catalog, **kwargs)
        except ValueError:
            raise ValueError(f"Cannot load catalog from '{catalog_to_load!s}', is it a catalog file?")

        if direct_load:
            if len(new_catalog) > 1:
                raise TypeError(f"Catalog cannot be direct loaded, and 'direct_load' is True")
            return new_catalog[0].function
        return new_catalog

    def __getattr__(self, key: str):
        result = getattr(Collection(*[ind for ind in self]), key)
        if len(result) == 1:
            return result[0]
        return result

    def __call__(self, *args, **kwargs):
        result = Collection(*[ind(*args, **kwargs) for ind in self])
        if len(result) == 1:
            return result[0]
        return result

    def __matmul__(self, other):
        return Collection(*[ind @ other for ind in self])

    def __getitem__(self, idx: int | str):
        """
        Allow programatic access to catalog entries via [] interface
        """
        if isinstance(idx, int):
            return self._catalog[list(self._catalog.keys())[idx]]

        if isinstance(idx, str) and idx in self._catalog:
            return self._catalog[idx]
        return self.__call__(idx)

    def items(self):
        for i in self._catalog.items():
            yield i

    def __len__(self):
        return len(self._catalog.keys())

    def __iter__(self):
        for key in self._catalog:
            yield self._catalog[key]

    def __str__(self):
        return f"{self.name if self.name else 'Data'} Catalog including {list(self._catalog.keys())}"

    @property
    def _doc_(self):
        return f"Catalog of Data"

    @property
    def _info_(self):
        return {cat._name_: cat._info_ for cat in self}

    ## Reprs
    def __repr__(self) -> str:
        if not UTILS_REPR:
            return str(self)

        return pyearthtools.utils.repr_utils.standard(
            *list(self),
            name=self.name or "Catalog",
            documentation_attr="_doc_",
            name_attr="_name_",
            info_attr="_info_",
        )

    def _repr_html_(self) -> str:
        if not UTILS_REPR:
            return repr(self)

        return pyearthtools.utils.repr_utils.html(
            *list(self),
            name=self.name or "Catalog",
            documentation_attr="_doc_",
            name_attr="_name_",
            info_attr="_info_",
        )

    ## Maths?
    def __add__(self, other):
        new_catalog = Catalog(name=self.name)
        new_catalog.append(self)
        new_catalog.append(other)
        return new_catalog

    def __radd__(self, other):
        new_catalog = Catalog(name=other.name)
        new_catalog.append(self)
        new_catalog.append(self)
        return new_catalog
