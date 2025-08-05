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


from pyearthtools.data.operations import binning
from pyearthtools.data.time import TimeDelta

import datetime
import xarray as xr
import numpy as np
import pytest


def test_binning():

    data = np.random.random((30, 30, 3))
    times = ["2020-01-01", "2020-01-02", "2020-01-03"]
    times = [datetime.datetime.fromisoformat(t) for t in times]

    da = xr.DataArray(coords={"lat": list(range(30)), "lon": list(range(30)), "time": times}, data=data)

    # Smoke tests
    binned = binning(da, "daily")
    assert binned is not None
    binned = binning(da, "daily", expand=False)
    assert binned is not None

    offset = TimeDelta(1, "days")
    binned = binning(da, "daily", offset=offset)
    assert binned is not None

    # Test exceptions
    with pytest.raises(ValueError):
        _binned = binning(da, "wobbly")

    with pytest.raises(AttributeError):
        _binned = binning(da, "daily", dimension="strange")
