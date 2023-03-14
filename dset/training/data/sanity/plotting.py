import functools
from typing import Any
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


from dset.training.data import DataIterator
from dset.training.data.sanity import iterator_retrieval


def _reduce_size(data, array_indexes: tuple[int]):
    if array_indexes is None:
        array_indexes = []

    count = 0
    while len(data.shape) > 2:
        index = array_indexes[count] if count < len(array_indexes) else 0
        data = data[index]
        count += 1
    return data


# @functools.lru_cache(10)
def _make_plot(ax, data: Any, array_indexes: list[int] = None, **plot_kwargs):
    if isinstance(array_indexes, tuple):
        array_indexes = list(array_indexes)
    if array_indexes is None:
        array_indexes = []
    array_indexes = [*array_indexes, *([0] * 20)]

    if isinstance(data, (tuple, list)):
        return _make_plot(ax, data[array_indexes.pop(0)], array_indexes, **plot_kwargs)

    if isinstance(data, xr.Dataset):
        data_var = list(data.data_vars)[array_indexes.pop(0)]
        return _make_plot(ax, data[data_var], array_indexes, **plot_kwargs)
    elif isinstance(data, xr.DataArray):
        data = _reduce_size(data, array_indexes)
        data.plot(ax=ax, **plot_kwargs)
    elif isinstance(data, (np.ndarray)):
        data = np.squeeze(data)
        data = _reduce_size(data, array_indexes)
        ax.imshow(data, **plot_kwargs)

    return ax


def plot(
    dataIterator: DataIterator,
    index: str = None,
    *,
    timeout: int = 20,
    array_indexes: tuple[int] = None,
    fig_kwargs: dict = {},
    **plot_kwargs
):
    result = iterator_retrieval.signal_data(
        dataIterator, idx=index, timeout=timeout, **plot_kwargs
    )

    fig, axes = plt.subplots(len(result.keys()), 1, **fig_kwargs)
    fig.suptitle("DataIterator Plots")

    for i, (iterator, data) in enumerate(result.items()):
        axes[i] = _make_plot(axes[i], [d for d in data][0], array_indexes, **plot_kwargs)
        axes[i].set_title(iterator_retrieval._get_iterator_name(iterator))

    fig.tight_layout(pad=5.0)
    return fig
