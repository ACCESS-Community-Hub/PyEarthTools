# Copyright Commonwealth of Australia, Bureau of Meteorology 2025.
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

from pyearthtools.pipeline.operations.numpy import conversion

import numpy as np
import xarray as xr


def test_ToXarray_with_DataArray():

    coords = {"x": list(range(5)), "y": list(range(5))}
    data = np.ones((5, 5))
    sample = xr.DataArray(coords=coords, data=data)

    tox = conversion.ToXarray.like(sample)
    result = tox.apply_func(data)

    assert (result == sample).all()

    as_numpy = tox.undo_func(sample)
    assert (as_numpy == data).all()


def test_ToXarray_with_Dataset():

    coords = {"x": list(range(5)), "y": list(range(5))}
    data = np.ones((5, 5))
    data1 = np.ones((1, 5, 5))
    sample_da = xr.DataArray(coords=coords, data=data)
    sample_ds = xr.Dataset(coords=coords, data_vars={"z": sample_da})

    tox = conversion.ToXarray.like(sample_ds)
    result = tox.apply_func(data1)

    assert (result == sample_ds).all()

    as_numpy = tox.undo_func(sample_ds)
    assert (as_numpy == data1).all()


def test_drop_coords():

    coords = {"x": list(range(5)), "y": list(range(5))}
    data = np.ones((5, 5))
    _data1 = np.ones((1, 5, 5))
    sample_da = xr.DataArray(coords=coords, data=data)
    sample_ds = xr.Dataset(coords=coords, data_vars={"z": sample_da})

    tox = conversion.ToXarray.like(sample_ds, drop_coords=["x"])
    assert tox is not None


def test_ToDask():

    data = np.ones((5, 5))

    tod = conversion.ToDask()
    da = tod.apply_func(data)
    orig = tod.undo_func(da)
    assert (orig == data).all()
