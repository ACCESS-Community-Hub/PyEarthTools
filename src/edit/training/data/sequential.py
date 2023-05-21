"""
Configure and Initialise a Data Pipeline

"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Callable
import yaml
import inspect

from edit.training.data.utils import get_pipeline, get_callable

from edit.training.data.templates import (
    DataIterator,
    DataInterface,
    DataStep,
    DataOperation,
)


def SequentialIterator(func: Callable) -> Callable:
    """
    Decorator to allow a Data Pipeline [Step][edit.training.data.templates.DataStep] to be partially initialised,
    such that the first element which will be the prior step, can be given later.


    Raises:
        RuntimeError: 
            If an attribute is requested from underlying object while it is uninitialised
        Any: 
            If an error is raised while initialising the underlying step

    Returns:
        (Callable): 
            Wrapper around underlying function
    """    


    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if args and len(args) > 0 and not isinstance(args[0], type):  
        # and isinstance(args[0], (DataIterator, DataInterface, DataStep, DataIndex, OperatorIndex)):
            return func(*args, **kwargs)

        if list(inspect.signature(func).parameters)[0] in kwargs.keys():
            return func(*args, **kwargs)

        class add_iterator:
            def __init__(self, func, *args, **kwargs):
                self.func = func
                self.args = args
                self.kwargs = kwargs

            def __getattr__(self, key):
                raise RuntimeError(
                    f"{self} cannot be used while it is waiting on an index."
                )

            def __call__(self, iterator: DataIterator):
                try:
                    if len(self.args) > 0 and isinstance(self.args[0], type):
                        return self.func(self.args[0], iterator, *self.args[1:], **self.kwargs)

                    return self.func(iterator, *self.args, **self.kwargs)
                except Exception as e:
                    raise type(e)(
                        f"Adding iterator to {func} raised an exception"
                    ) from e

            def __repr__(self) -> str:
                return (
                    f"Sequential Iterator for {func.__name__} waiting on an index."
                )

        return add_iterator(func, *args, **kwargs)

    return wrapper


def Sequential(*args: list[DataStep]) -> DataStep:
    """
    Combine partially initialised [DataStep's][edit.training.data.templates.DataStep]
    to form a fully defined Pipeline

    !!! Example
        ```python
        Sequential(
            DataStep(variable = True)   #Only Partially Configured
            AnotherStep()               #Only Partially Configured
        )

        ```
    
    Args:
        *args (Any):
            DataIterator - with @SequentialIterator
    Returns:
        (DataStep): 
            Fully initialised DataStep's aka a Data Pipeline
    """    
    iterator = args[0]
    for i in range(1, len(args)):
        iterator = args[i](iterator)
    return iterator


def from_dict(data_specifications: dict | str | Path ) -> DataStep:
    """
    Create a Data Pipeline from a dictionary, or a yaml file.

    Use class names as keys, will auto try [edit.training.data][edit.training.data].KEY, or fail over onto looking on the Python PATH.

    Values inside the dictionary are expected to also be a dictionary with elements to be passed as keyword arguments to initalise the [DataStep][edit.training.data.templates.DataStep]

    !!! "tip" Notes
        * Specify `order` to set order
        * If using two of the same [DataStep][edit.training.data.templates.DataStep], add [NUMBER] to indicate,

    ??? Example
        ```python
        data_pipeline = {
            "DataStep": {'variable': True},
            "AnotherStep": {}
        }
        from_dict(data_pipeline)
        ```


    Args:
        data_specifications (dict | str | Path ): 
            Dictionary object or path to file specifying the Data Pipeline

    Returns:
        (DataStep): 
            Fully initialised DataStep's aka a Data Pipeline
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

    data_list = get_pipeline(data_specifications, order)
    return Sequential(*data_list)
