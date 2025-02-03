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


from __future__ import annotations
from typing import Any

import xarray as xr
from xarray import plot as xplt

import numpy as np

from matplotlib.figure import Figure

from pyearthtools.data.indexes import FileSystemIndex
from pyearthtools.data.save import dataset, json, array, plot

DASK_IMPORTED = True
try:
    import dask
    from dask.delayed import Delayed
    import dask.array as da
    from pyearthtools.data.save import dask as dask_save
except (ImportError, ModuleNotFoundError) as e:
    DASK_IMPORTED = False

DISPATCH = {
    dataset: [xr.Dataset, xr.DataArray],
    json: [dict],
    array: [np.ndarray],
    plot: [xplt.FacetGrid, Figure],
}

if DASK_IMPORTED:
    DISPATCH[dask_save] = [da.Array]

ATTRIBUTE_SEARCH = {plot: ["savefig"]}


def invert_dictionary_list(dictionary: dict) -> dict:
    return_dict = {}
    for key, value in dictionary.items():
        for item in value:
            return_dict[item] = key
    return return_dict


DISPATCH = invert_dictionary_list(DISPATCH)
ATTRIBUTE_SEARCH = invert_dictionary_list(ATTRIBUTE_SEARCH)


def save(data: Any, callback: FileSystemIndex, *args, save_kwargs: dict = {}, **kwargs):
    """
    Save data at location specified by an Index

    Automatically inferes to how to save data based on the type

    Uses args and kwargs in `callback.search` to find path

    Args:
        data (Any):
            Data to be saved
        callback (FileSystemIndex):
            FileSystemIndex to use to discover where to save data
        *args (Any, optional):
            Arguments to be passed to `callback.search` to find file path
        save_kwargs (dict, optional):
            Kwargs to pass to underlying save function
        *kwargs (Any, optional):
            Keyword arguments to be passed to `callback.search` to find file path

    Raises:
        TypeError:
            If type that is not known is passed

    Returns:
        (Path):
            Location where data was saved
    """

    if not isinstance(callback, FileSystemIndex):
        raise TypeError("Data cannot be saved without a 'FileSystemIndex'")
    type_search = type(data)

    if isinstance(data, (tuple, list)):
        type_search = type(data[0])

    if DASK_IMPORTED and type_search == Delayed:
        if isinstance(data, (tuple, list)):
            data = dask.compute(data)
        else:
            data = data.compute()
        return save(data, callback, *args, save_kwargs=save_kwargs, **kwargs)

    dispatch_dict = DISPATCH

    if type_search not in DISPATCH:
        for attr, code in ATTRIBUTE_SEARCH.items():
            if hasattr(data, attr):
                return code.save(data, callback, *args, **kwargs)

        raise TypeError(f"Unable to auto save data of {type_search}" f"\nAllowed types are: {list(DISPATCH.keys())}")

    return dispatch_dict[type_search].save(data, callback, *args, save_kwargs=save_kwargs, **kwargs)
