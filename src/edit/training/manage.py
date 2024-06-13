"""
Variable management

"""

from __future__ import annotations
from typing import Any, overload, Union, Optional, Callable, TypeVar

import xarray as xr
import numpy as np
import torch
import itertools


DATA_TYPES = Union[np.ndarray, xr.Dataset, torch.Tensor, list]
D = TypeVar("D", np.ndarray, xr.Dataset, torch.Tensor, list)


COMBINE_FUNCTIONS: dict[type, Callable[[tuple], Any]] = {
    np.ndarray: np.concatenate,
    xr.Dataset: xr.merge,
    torch.Tensor: torch.concat,
    list: (lambda x: itertools.chain(*x)),
}


class Variables:
    """
    Variable management class.

    Allows for the specification of categories of variables in an array, and the ordering.

    Provides functions for:
        - Reordering
        - Splitting
        - Adding
        - Removing
        - Extracting
        - Joining

    These functions can be run on any incoming data, with a choice of the order associated.

    The order is specified as the first letter of the category capitalised.

    All arrays are expected to be channel / variable first.

    E.g.
    >>> Variables(order = 'PFD', prognostics = 10, diagnostics = 5, forcings = 7)
        Variables - (PFD)
        prognostics - 10
        forcings - 7
        diagnostics - 5

    >>> variables = Variables(order = 'PFD', prognostics = 10, diagnostics = 5, forcings = 7)
    >>> variables.split(data)
        {
            # Categories of data split accordingly
        }
    >>> variables.extract(data, 'diagnostics')
        # diagnostics extracted from the data
    """

    order: str
    categories: dict[str, list[str] | int]

    def __init__(
        self,
        variables: Optional[Union[Variables, dict[str, list[str] | int]]] = None,
        *,
        order: Optional[str] = None,
        **kwargs,
    ):
        """
        Setup Variable Manager

        Args:
            order (Optional[str], optional):
                Order of categories.
                Uses the first letter capitalised.

                If not given, is inferred from order of `kwargs`
        """

        if variables:
            if isinstance(variables, Variables):
                variables = variables.categories
            kwargs.update(variables)

        if order is None:
            order = "".join(str(k)[0].upper() for k in kwargs.keys())

        self.order = order.upper()
        if not all((k[0].upper() in order for k in kwargs)):
            raise ValueError(
                f"Not all categories represented in the order. {list(kwargs.keys())!s} cannot work with {order!r}."
            )

        self.categories = kwargs

    @property
    def category_names(self) -> list[str]:
        """Get names of categories specified, may not be in order."""
        return list(self.categories.keys())

    def check_category(self, category: str):
        """Check category, to see if it is specified"""
        if category not in self.categories:
            raise ValueError(
                f"Attempting to modify/view category: {category!r} when it was not included. {self.category_names}."
            )
        return

    def check_order(self, order: str):
        """Check order to see if all elements are valid"""
        for o in order:
            if o not in self.order:
                raise ValueError(f"Discovered {o} in `order` which was not in the specified order: {self.order}.")

    def __getattr__(self, key):
        if key in self.categories:
            return self.categories[key]
        raise AttributeError(f"{self.__class__.__name__!s} has no attribute {key!r}.")

    def names_from_order(self, order: Optional[str] = None, reorder: bool = False) -> tuple[str, ...]:
        """
        Get names of categories from `order`

        Args:
            order (Optional[str], optional):
                Order to get names from. Defaults to None.
            reorder (bool, optional):
                Whether to reorder `order` to be in specified `init` order. Defaults to False.

        Returns:
            (tuple[str, ...]):
                Names of categories in `order` or if `reorder` 'correct' order.
        """

        order_to_name = {n[0].upper(): n for n in self.category_names}

        order = order.upper() if order else self.order
        self.check_order(order)

        if reorder:
            order = "".join(o for o in self.order if o in order)
        return tuple(order_to_name[o] for o in order)

    def np_slices(self, order: Optional[str] = None) -> dict[str, slice]:
        """
        Get slice for extraction of data

        Args:
            order (Optional[str], optional):
                Incoming order if different from specified. Defaults to None.

        Returns:
            (dict[str, slice]):
                Category to slice pairs
        """
        """Get slices for data extracted based on incoming order"""
        names: tuple[str, ...] = self.names_from_order(order=order)
        given_vars = tuple(getattr(self, x) for x in names)

        if not (
            not all(map(lambda x: isinstance(x, int), given_vars))
            or not all(map(lambda x: isinstance(x, list), given_vars))
        ):
            raise TypeError(f"All variables must be one type either, int or list[str] not both.")

        if all(map(lambda x: isinstance(x, list), given_vars)):
            return {
                names[x]: slice(sum(map(len, given_vars[:x])) if x > 0 else 0, sum(map(len, given_vars[: x + 1])))
                for x in range(len(given_vars))
            }

        return {
            names[x]: slice(sum(given_vars[:x]) if x > 0 else 0, sum(given_vars[: x + 1]))
            for x in range(len(given_vars))
        }

    def xr_slices(self, data: xr.Dataset, order: Optional[str] = None) -> dict[str, list[str]]:
        """
        Get slices for use in xarray.

        Return lists of variable names

        Requires data in case variables are given as int
        """

        list_vars = list(data.data_vars)

        names: tuple[str, ...] = self.names_from_order()
        slices = self.np_slices(order=order)

        return {name: list_vars[slices[name]] if isinstance(self.categories[name], int) else self.categories[name] for name in names}  # type: ignore

    def reorder(self, data: D, order: Optional[str] = None) -> D:
        """
        Reorder incoming data into the originally specified order
        """
        self.compare_length(data, order=order, error=True)

        if isinstance(data, xr.Dataset):
            return data[list(self.names_from_order())]

        if order is None:
            raise TypeError(f"Order must be given for arrays.")

        slices = self.np_slices(order)
        data_dict = {key: data[val] for key, val in slices.items()}
        return COMBINE_FUNCTIONS[type(data)](tuple(data_dict[na] for na in self.names_from_order()))

    def join(self, **kwargs: DATA_TYPES) -> DATA_TYPES:
        """
        Join data given in `kwargs`, will be ordered based on specified order
        """
        kwarg_names = list(kwargs.keys())
        order = "".join(str(n)[0].upper() for n in kwarg_names)
        self.check_order(order)

        for key, val in kwargs.items():
            if not self.compare_length(val, order=key[0].upper()):
                raise ValueError(f"Data for {key!s} has incorrect length, expected {self.length(key)}, got {len(val)}.")

        if not all(type(kwargs[kwarg_names[0]]) == type(kwargs[kwarg_names[i]]) for i in range(1, len(kwarg_names))):
            raise TypeError(f"All given data must be of the same type.")

        dtype = type(kwargs[kwarg_names[0]])
        return COMBINE_FUNCTIONS[dtype](tuple(kwargs[key] for key in self.names_from_order(order, reorder=True)))

    def split(self, data: D, order: Optional[str] = None) -> dict[str, D]:  # type: ignore
        """
        Split incoming data into the specified categories

        Args:
            data (DATA_TYPES):
                Data to split
            order (Optional[str], optional):
                Order of data, required if data is an array. Defaults to None.

        Returns:
            (dict[str, DATA_TYPES]):
                Data split, with keys based on specified
        """
        self.compare_length(data, order=order, error=True)

        if isinstance(data, xr.Dataset):
            return {key: data[val] for key, val in self.xr_slices(data, order=order).items()}
        return {key: data[val] for key, val in self.np_slices(order=order).items()}

    @overload
    def add(self, data: D, incoming: tuple[D], category: tuple[str], order: Optional[str] = None) -> D: ...

    @overload
    def add(self, data: D, incoming: D, category: str, order: Optional[str] = None) -> D: ...

    def add(self, data: D, incoming: tuple[D] | D, category: tuple[str] | str, order: Optional[str] = None) -> D:
        """
        Add `incoming` data into the `data`, in the correct spot.

        Result will be reorded into the 'correct' order.

        See `.join` for explict assignment.

        Args:
            data (DATA_TYPES):
                Data to add to
            incoming (DATA_TYPES):
                Data to add
            category (str):
                Category of data to add
            order (Optional[str], optional):
                Order of `data` if different, expects no entry for `category`.
                Defaults to None.

        Returns:
            (DATA_TYPES):
                Merged data in the order as specified in `init`.

        Examples:
            >>> vars = Variables(order = 'PFD', prognostics = 10, diagnostics = 5, forcings = 7)
            >>> forcings_missing = np.ones((15))
            >>> forcings = np.zeros((7))
            >>> data = vars.add(forcings_missing, forcings, order = 'PD')
            >>> data.shape
                (22,)
        """
        order = (order or self.order.replace(category[0].upper(), "")).upper()
        self.check_order(order)

        if isinstance(incoming, tuple) or isinstance(category, tuple):
            if isinstance(incoming, tuple) ^ isinstance(category, tuple):
                raise TypeError(f"If `incoming` or `category` is tuple, both must be.")

            assert isinstance(incoming, tuple) and isinstance(category, tuple)  # Type hint assistance

            for i, c in zip(incoming, category):
                data = self.add(data, i, c, order=order)
                order = "".join(o for o in self.order if o in (*order, c[0].upper()))
            return data

        self.compare_length(data, order=order, error=True)
        self.compare_length(incoming, order=category[0].upper(), error=True)
        self.check_category(category)

        slices = self.np_slices(order=order)

        if not type(data) == type(incoming):
            raise TypeError(f"`data` and `incoming` must be of the same type, not {type(data)} and {type(incoming)}.")

        if isinstance(data, xr.Dataset) and isinstance(incoming, xr.Dataset):
            return xr.merge((data, incoming))[list(self.names_from_order())]

        data_dict = {key: data[val] for key, val in slices.items()}  # type: ignore
        data_dict[category] = incoming

        return COMBINE_FUNCTIONS[type(data)](tuple(data_dict[na] for na in self.names_from_order() if na in data_dict))

    def remove(self, data: D, category: str | tuple[str], order: Optional[str] = None) -> D:
        """
        Remove a category of data.

        If `order` is given, `data` can contain only a subset of categories,
        and will return in the given order.

        Args:
            data (DATA_TYPES):
                Data to remove category from
            category (str):
                Category to remove
            order (Optional[str], optional):
                Order if different to specified. Can be subset.
                Defaults to None.

        Returns:
            (DATA_TYPES):
                `data` with `category` removed.
                If `order` given will be maintained.
        """

        if isinstance(category, tuple):
            order = (order or self.order).upper()

            for cat in category:
                data = self.remove(data, category=cat, order=order)
                order = order.replace(cat[0].upper(), "")
            return data

        self.compare_length(data, order=order, error=True)
        self.check_category(category)

        order = order or self.order
        self.check_order(order)

        slices = self.np_slices(order=order)

        removed_order = tuple(i for i in self.names_from_order(order=order) if not i == category)

        if isinstance(data, xr.Dataset):
            xr_slices = self.xr_slices(data)
            return xr.merge(tuple(data[xr_slices[na]] for na in removed_order))

        data_dict = {key: data[val] for key, val in slices.items()}
        return COMBINE_FUNCTIONS[type(data)](tuple(data_dict[na] for na in removed_order))

    @overload
    def extract(self, data: D, category: tuple[str, ...], order: Optional[str] = None) -> tuple[D, ...]: ...

    @overload
    def extract(self, data: D, category: str, order: Optional[str] = None) -> D: ...

    def extract(self, data: D, category: tuple[str, ...] | str, order: Optional[str] = None) -> D | tuple[D, ...]:
        """
        Extract a category from the `data`.

        If `order` is given, `data` can contain only a subset of categories,
        and will extract correctly.

        Args:
            data (DATA_TYPES):
                Data to extract category from.
            category (tuple[str, ...] | str):
                Category to extract
            order (Optional[str], optional):
                Order if different to specified. Defaults to None.

        Returns:
            (tuple[DATA_TYPES, ...] | DATA_TYPES):
                Extracted data
        """
        if isinstance(category, tuple):
            return tuple(map(lambda cat: self.extract(data, category=cat, order=order), category))

        self.compare_length(data, order=order, error=True)
        self.check_category(category)

        order = order or self.order
        self.check_order(order)

        slices = self.np_slices(order=order)

        if isinstance(data, xr.Dataset):
            xr_slices = self.xr_slices(data)
            return data[xr_slices[category]]

        data_dict = {key: data[val] for key, val in slices.items()}

        return data_dict[category]

    def __repr__(self) -> str:
        """Repr function"""

        def spacing(k, v, spacing=20):
            return f"{k}{''.join([' '] * (spacing - len(k)))}{v!r}"

        category_info = "\n".join("\t" + spacing(n, self.categories[n]) for n in self.names_from_order())

        return f"{super().__repr__()}\n" f"Variables - ({self.order}):\n" f"{category_info}"

    def length_of_order(self, order: str) -> int:
        """
        Get expected length of categories as listed in `order`.
        """
        self.check_order(order)

        names = self.names_from_order(order=order)
        return sum(tuple(self.length(x) for x in names))

    def compare_length(self, data: DATA_TYPES, order: Optional[str] = None, error: bool = False) -> bool:
        """
        Compare length of data to that of expected.

        Raise error if `error`.
        """
        length = self.length_of_order(order or self.order)
        if isinstance(data, (np.ndarray, torch.Tensor)):
            data_length = data.shape[0]
        elif isinstance(data, xr.Dataset):
            data_length = len(list(data.data_vars))
        elif hasattr(data, "__len__"):
            data_length = len(data)
        else:
            raise TypeError(f"Cannot find length of {data}.")

        if not error:
            return length == data_length
        elif not length == data_length:
            raise ValueError(
                f"Expected length and data length differed. {length} != {data_length}. \nExpected order {order or self.order} - {self.categories}."
            )
        return True

    def length(self, category: str | list[str] | tuple[str, ...]) -> int:
        """
        Get length of categories
        """
        if isinstance(category, (tuple, list)):
            return sum(map(self.length, category))

        self.check_category(category)

        cat = self.categories[category]
        return len(cat) if isinstance(cat, list) else cat

    def __len__(self):
        return sum(tuple(self.length(x) for x in self.category_names))

    def __contains__(self, key: str) -> bool:
        return key in self.categories
