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

from pyearthtools.pipeline.operations.xarray import conversion

import numpy as np
import xarray as xr
import pytest


def test_exceptions():

    with pytest.raises(ValueError):
        _ton = conversion.ToNumpy(reference_dataset=True, saved_records=True)


def test_Numpy():

    coords = {"x": list(range(5)), "y": list(range(5))}
    data = np.ones((5, 5))
    sample = xr.DataArray(coords=coords, data=data)

    ton = conversion.ToNumpy()
    result = ton.apply_func(sample)
    assert (result == data).all()


def test_ToDask():

    coords = {"x": list(range(5)), "y": list(range(5))}
    data = np.ones((5, 5))
    sample = xr.DataArray(coords=coords, data=data)

    tod = conversion.ToDask()
    _da = tod.apply_func(sample)
    # orig = tod.undo_func(da)
    # assert (orig == sample).all()
