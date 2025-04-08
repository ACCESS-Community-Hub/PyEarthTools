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

from pyearthtools.data.operations import percentile

import xarray as xr
import numpy as np


def test_percentile():

    data = np.linspace(0, 100, 100)
    da = xr.DataArray(coords={"index": list(range(0, 100))}, data=data, name="Sam")

    ds = xr.Dataset(coords={"index": list(range(0, 100))}, data_vars={"temp": da})

    result = percentile(ds, [10, 90])
    np.testing.assert_allclose(result["temp"].values, (10, 90))

    result = percentile(ds, 10)
    np.testing.assert_allclose(result["temp"].values, (10,))

    result = percentile(da, [10, 90])
