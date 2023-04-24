import functools
import math
from typing import Any, Union
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


from dset.data import Collection

from dset.training.data import DataIterator
from dset.training.data.sanity import iterator_retrieval


def _reduce_size(data, array_indexes: tuple[int]):
    while len(data.shape) > 2:
        data = data[array_indexes.pop()]
    return data, array_indexes


def _make_plot(ax, data: Any, array_indexes: list[int] = None, new_index = True, **plot_kwargs):
    if array_indexes is None:
        array_indexes = []
        
    if isinstance(array_indexes, tuple):
        array_indexes = list(array_indexes)
    elif not isinstance(array_indexes, list):
        array_indexes = [array_indexes]

    if new_index:
        array_indexes = [*array_indexes, *([0] * 20)]
        array_indexes.reverse()

    shape = (None)

    if isinstance(data, (tuple, list, Collection)):
        ax, shape = _make_plot(ax, data[array_indexes.pop()], array_indexes, new_index = False, **plot_kwargs)
        return ax, (len(data), *shape)

    if isinstance(data, xr.Dataset):
        data_var = list(data.data_vars)[array_indexes.pop()]
        ax, shape =  _make_plot(ax, data[data_var], array_indexes, new_index = False, **plot_kwargs)
        return ax, (len(list(data.data_vars)), *shape)

    elif isinstance(data, xr.DataArray):
        shape = data.shape
        data, array_indexes = _reduce_size(data, array_indexes)
        data.plot(ax=ax, **plot_kwargs)

    elif isinstance(data, (np.ndarray)):
        shape = data.shape
        #data = np.squeeze(data)
        data, array_indexes = _reduce_size(data, array_indexes)
        im = ax.imshow(data, **plot_kwargs)
        plt.colorbar(im, ax=ax)

    elif isinstance(data, str):
        return ax, data

    return ax, shape


def plot(
    dataIterator: DataIterator,
    index: str = None,
    *,
    timeout: int = 20,
    array_indexes: dict[str, list[int]] | list[list[int]] = None,
    fig_kwargs: dict = {'figsize': (25,20)},
    layout_kwargs: dict = {'pad': 5},
    **plot_kwargs
):
    """
    Plot a given Data Pipline. 

    Each step will be plotted individually, with the shape also shown.
    If index is not given, steps below an Iterator will fail, as data cannot be retrieved by iterating.

    Parameters
    ----------
    dataIterator
        Data Pipeline to plot
    index, optional
        Date index to plot at, by default None
    timeout, optional
        Time allowed for data to be retrieved, by default 20
    array_indexes, optional
        Indexes to be passed to flattening function, 
        Can be list, where position refers to index, or dict where key == name
        Each element corresponds to an individual step, by default None
    fig_kwargs, optional
        Other kwargs for fig creation, by default {'figsize': (25,20)}
    layout_kwargs, optional
        Kwargs to be passed to plt.tight_layout, by default {'pad': 5}

    Returns
    -------
        Figure
    """
    result = iterator_retrieval.signal_data(
        dataIterator, idx=index, timeout=timeout)

    num_iterator = len(result.keys())
    size = math.ceil(math.sqrt(num_iterator))

    fig, axes = plt.subplots(size, size, **fig_kwargs)
    fig.suptitle("Plots of each step in Data Pipeline. (->)")
    plt.tight_layout(**layout_kwargs)

    if not array_indexes:
        array_indexes = [[0]] * len(result.keys())
    if isinstance(array_indexes, list):
        array_indexes = [*array_indexes, *([[0]] * (len(result.keys())-len(array_indexes)))]

    for i, (iterator, data) in enumerate(result.items()):
        name = iterator_retrieval._get_iterator_name(iterator)
        if isinstance(array_indexes, list):
            indexes = array_indexes[i]
        elif isinstance(array_indexes, dict) and name in array_indexes:
            indexes = array_indexes[name]
        else:
            indexes = None

        coords = (i // size, i % size)
        axes[coords[0], coords[1]], info = _make_plot(axes[coords[0], coords[1]], data, indexes, **plot_kwargs)
        axes[coords[0], coords[1]].set_title(f"{name}.\n{info}")
    
    while i < (size * size)-1:
        i += 1
        coords = (i // size, i % size)
        fig.delaxes(axes[coords[0], coords[1]])


    return fig
