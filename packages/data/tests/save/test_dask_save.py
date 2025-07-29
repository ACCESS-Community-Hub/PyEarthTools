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

from pyearthtools.data.save import dask as dasksave

import xarray as xr
import numpy as np


def test_save(monkeypatch):
    def mock_save(da, *args, **kwargs):
        return da

    monkeypatch.setattr(dasksave, "numpy_save", mock_save)

    da = xr.DataArray(np.ones((50, 50)))
    assert len(da.chunksizes) == 0

    da2 = da.chunk(10)
    assert len(da2.chunksizes["dim_0"]) == 5

    # Smoke test the save operation which just calls compute
    dasksave.save(da2)
