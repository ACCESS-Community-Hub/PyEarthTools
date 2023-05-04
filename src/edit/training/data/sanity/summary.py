from __future__ import annotations

from typing import Any
import numpy as np
import xarray as xr


from edit.training.data import DataIterator, DataStep
from edit.training.data.sanity import iterator_retrieval


def _get_data_info(data: Any, index: int = 0):
    if isinstance(data, (list, tuple)):
        return _get_data_info(data[index])
    if hasattr(data, "shape"):
        return str(data.shape)
    elif isinstance(data, (xr.DataArray, xr.Dataset)):
        return str(data)
    elif isinstance(data, str):
        return data
    else:
        return f"{type(data)}"


def _format(iterator, value, padding_size=30, tab_size=10, index: int = 0):
    def padding(name, value: tuple[str], padding_size=30, tab_size=10):
        padding_create = lambda a: a + "".join([" "] * (padding_size - len(a)))
        tab_size = "".join([" "] * tab_size)

        return_string = tab_size + padding_create(name)

        if "\n" in value:
            initial_size = "".join([" "] * len(return_string))
            value = "\n".join(
                [
                    ("" if i == 0 else initial_size) + line
                    for i, line in enumerate(value.split("\n"))
                ]
            )

        return return_string + value

    name = iterator_retrieval._get_iterator_name(iterator)
    value_info = _get_data_info(value, index=index)

    return padding(name, value_info, padding_size=padding_size, tab_size=tab_size)


def summary(
    dataIterator: DataIterator | DataStep,
    index: str = None,
    *,
    timeout: int = 20,
    tuple_index: int = 0,
    verbose: bool = True,
) -> str:
    """
    Create printed summary of a Data Pipeline, with shape and type shown

    !!! Warning
        If index is not given, steps below an Iterator will fail, as data cannot be retrieved by iterating.

    Args:
        dataIterator (DataIterator): 
            Data Pipeline to summarise
        index (str, optional): 
            Date index to plot at. Defaults to None.
        timeout (int, optional): 
            Time allowed for data to be retrieved. Defaults to 20.
        tuple_index (int, optional): 
            If data is a tuple, which element to use. Defaults to 0.
        verbose (bool, optional):
            Whether to print summary. Defaults to True

    Returns:
        (str):
            Summary as a string

    """    
    result = iterator_retrieval.signal_data(dataIterator, idx=index, timeout=timeout)

    str_summary = ""
    def add_summary(text: str):
        if verbose:
            print(text)
        str_summary += text + '\n'

    add_summary("\n---- Summary for Data Iterator ----")
    add_summary(
        "\nNOTE: Normalisation Calculations can take significant amounts of time.\nEnsure that it has been safely run before this.\n"
    )
    for iterator, values in result.items():
        add_summary(_format(iterator, [v for v in values][0], index=tuple_index))

    return str_summary