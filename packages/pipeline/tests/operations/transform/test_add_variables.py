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

from pyearthtools.pipeline.operations.transform import add_variables

import datetime
import xarray as xr
import numpy as np
import pytest


def test_TimeOfYear():

    times = [datetime.datetime(2020, 1, 1) + datetime.timedelta(days=day) for day in range(365)]

    da = xr.DataArray(coords={"time": times, "level": [1, 2]}, data=np.ones((365, 2)))

    ds = xr.Dataset(coords={"time": times, "level": [1, 2]}, data_vars={"temperature": da})

    toy = add_variables.TimeOfYear("dayofyear")
    _result = toy.apply(ds)
    assert toy._info_ == {"method": "dayofyear"}

    toy = add_variables.TimeOfYear("monthofyear")
    _result = toy.apply(ds)
    assert toy._info_ == {"method": "monthofyear"}

    with pytest.raises(ValueError):
        _toy = add_variables.TimeOfYear("century")
