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

from pyearthtools.data.transforms import coordinates
import xarray as xr
import numpy as np


# Create a 2D array of integers
data = np.array([[0, 1], [0, 1]], dtype=np.int32)

# Create a DataArray with named dimensions and coordinates
da = xr.DataArray(
    data,
    dims=["x", "y"],
    coords={"x": [10, 11], "y": [100, 110]},
    name="example_integers"
)


larger_data = np.array(
    [
        [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3]
        ],
        [
            [0, 1, 2, 3],
            [1, 16, 17, 18],
            [2, 110, 111, 112]
        ],
        [
            [0, 1, 2, 3],
            [1, 6, 7, 8],
            [2, 10, 11, 12]
        ],
        [
            [0, 1, 2, 3],
            [1, 16, 17, 18],
            [2, 110, 111, 112]
        ],
    ],
    dtype=np.int32)

da_larger = xr.DataArray(
    larger_data,
    dims=["height", "lat", "lon"],
    coords={"height": [10, 11, 12, 13], "lat": [100, 101, 102], "lon": [21, 22, 23, 24]},
    name="sample_data"
)


def test_Flatten():
    f = coordinates.Flatten(["height"])
    f.apply(da_larger)
