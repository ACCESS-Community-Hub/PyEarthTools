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


from pyearthtools.pipeline.operations.dask.compute import Compute

import xarray as xr
import numpy as np


def test_Compute():

    c = Compute()

    data = np.ones((50, 50))
    da = xr.DataArray(coords={"a": list(range(0, 50)), "b": list(range(0, 50))}, data=data)

    da.chunk(10)

    computed = c.apply_func(da)
    assert len(computed.chunksizes) == 0

    # Nonchunked things should be fine
    # It's not so much that this is a numpy array, just that I know it doesn't
    # have a compute attribute which is hard to guarantee for xarrays when dask
    # is installed
    _computed2 = c.apply_func(data)
