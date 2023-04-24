import functools
from abc import abstractmethod
from typing import Callable, Union

import yaml
import inspect
from datetime import datetime



from dset.data import DataIndex, OperatorIndex
from dset.data.time import DSETDatetime, time_delta
from dset.training.data.utils import get_indexes, get_callable

from dset.training.data.templates import DataIterator, DataInterface, DataStep

def SequentialIterator(func):
    """
    Decorator to allow Iterator's to not be fully specified,
    such that the first element of a (DataIterator, DataInterface, DataIndex, OperatorIndex) is missing.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if args:# and isinstance(args[0], (DataIterator, DataInterface, DataStep, DataIndex, OperatorIndex)):
            return func(*args, **kwargs)

        if list(inspect.signature(func).parameters)[0] in kwargs.keys():
            return func(*args, **kwargs)

        class add_iterator:
            def __init__(self, func, *args, **kwargs):
                self.func = func
                self.args = args
                self.kwargs = kwargs

            def __getattr__(self, key):
                raise RuntimeError(f"{self} cannot be used while it is waiting on an iterator.")

            def __call__(self, iterator: DataIterator):
                try:
                    return self.func(iterator, *self.args, **self.kwargs)
                except Exception as e:
                    raise type(e)(f"Adding iterator to {func} raised an exception") from e
            
            def __repr__(self) -> str:
                return f"Sequential Iterator for {func.__name__} waiting on an iterator."

        return add_iterator(func, *args, **kwargs)

    return wrapper


def Sequential(*args: list["DataIterator"]) -> "DataIterator":
    """
    From a list of DataIterators missing only an iterator,
    build a full DataIterator

    *args
        DataIterator - with @SequentialIterator

    Returns
    -------
        DataIterator
    """
    iterator = args[0]
    for i in range(1, len(args)):
        iterator = args[i](iterator)
    return iterator


def from_dict(data_specifications: str | dict) -> "DataIterator":
    """
    Create DataIterator from a dictionary.

    Use keys as class names, if not found will auto try dset.training.data.~

    Specify order to set order

    Parameters
    ----------
    data_specifications
        Dictionary containg Data Specifications. Can also be path to yaml file

    Returns
    -------
        DataIterator

    Raises
    ------
    TypeError
        If imported class cannot be understood
    """
    
    if isinstance(data_specifications, str):
        with open(data_specifications, "r") as file:
            data_specifications = yaml.safe_load(file)
        if "data" in data_specifications:
            data_specifications = data_specifications["data"]

    data_specifications = dict(**data_specifications)

    if "order" in data_specifications:
        order = data_specifications.pop("order")
    else:
        order = list(data_specifications.keys())

    data_list = get_indexes(data_specifications, order)
    return Sequential(*data_list)