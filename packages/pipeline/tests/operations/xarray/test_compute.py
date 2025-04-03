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

from pyearthtools.pipeline.operations.xarray import compute

import xarray as xr
import numpy as np


def test_compute():

    compute_operation = compute.Compute()

    da = xr.DataArray(np.ones((50, 50)))
    da2 = da.chunk(10)

    # Test computation
    da3 = compute_operation.apply_func(da2)
    assert len(da3.chunksizes) == 0

    # Test non-computation
    result = compute_operation.apply_func("hello")
    assert result == "hello"
