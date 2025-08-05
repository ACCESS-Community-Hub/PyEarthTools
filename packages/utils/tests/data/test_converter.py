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

SIMPLE_DATA_ARRAY = xr.DataArray([1, 2, 3, 4, 5])
SIMPLE_DATA_SET = xr.Dataset({"Entry": SIMPLE_DATA_ARRAY})


def test_NumpyConverter():
    """
    This test provides coverage, but does not test for
    correctness
    """

    # This round-trips convert and unconvert
    nc = converter.NumpyConverter()
    _np_array1 = nc.convert_from_xarray(SIMPLE_DATA_ARRAY)

    # FIXME
    # xr_da1 = nc.convert_to_xarray(np_array1)

    # Test conversion from xarray works
    nc = converter.NumpyConverter()
    _np_array2 = nc.convert_from_xarray(SIMPLE_DATA_SET)


def test_DaskConverter():
    """
    This test provides coverage, but does not test for
    correctness
    """

    dc = converter.DaskConverter()

    _da_array1 = dc.convert_from_xarray(SIMPLE_DATA_ARRAY)

    # FIXME
    # xr_da1 = dc.convert_to_xarray(da_array1)
