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

from pyearthtools.data.operations.utils import identify_time_dimension as itd

import xarray as xr
import numpy as np
from datetime import datetime as dt


def test_identify_time_dimension():

    times = ["2020-01-01", "2020-01-02"]
    datetimes = [dt(2020, 1, 1), dt(2020, 1, 2)]
    data = np.linspace(0, 10, 2)

    # Test array without a time dim
    da = xr.DataArray(coords={"rabbits": times}, data=data)

    result = itd(da)
    assert result == "time"

    # Test time named time
    da = xr.DataArray(coords={"time": times}, data=data)

    result = itd(da)
    assert result == "time"

    # Test time named time
    da = xr.DataArray(coords={"poodletime": times}, data=data)

    result = itd(da)
    assert result == "poodletime"

    # Test an unusual name but actually it's times
    da = xr.DataArray(coords={"something": datetimes}, data=data)

    result = itd(da)
    assert result == "something"

    result = itd(da)
