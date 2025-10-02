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


import xarray as xr

from pyearthtools.utils.data import converter

SIMPLE_DATA_ARRAY = xr.DataArray([1, 2, 3, 4, 5], dims=("x",), coords={"x": [0, 1, 2, 3, 4]}, name="Entry")
SIMPLE_DATA_SET = xr.Dataset({"Entry": SIMPLE_DATA_ARRAY})


def test_NumpyConverter():
    """
    Checks conversion from xarray to numpy and back
    """

    nc_da = converter.NumpyConverter()
    _ = nc_da.convert_from_xarray(SIMPLE_DATA_ARRAY)

    nc = converter.NumpyConverter()
    np_array = nc.convert_from_xarray(SIMPLE_DATA_SET)
    xr_ds = nc.convert_to_xarray(np_array)
    assert isinstance(xr_ds, xr.Dataset)
    assert "Entry" in xr_ds
    xr.testing.assert_identical(xr_ds["Entry"], SIMPLE_DATA_ARRAY)


def test_DaskConverter():
    """
    Checks conversion from xarray to dask and back
    """

    dc = converter.DaskConverter()

    da_array = dc.convert_from_xarray(SIMPLE_DATA_SET)
    da_array = da_array.compute()
    xr_ds = dc.convert_to_xarray(da_array)
    assert isinstance(xr_ds, xr.Dataset)
    assert "Entry" in xr_ds
    xr.testing.assert_identical(xr_ds["Entry"], SIMPLE_DATA_ARRAY)
