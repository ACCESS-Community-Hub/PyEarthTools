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


import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pyearthtools.utils.data import converter

SIMPLE_DATA_ARRAY = xr.DataArray([1, 2, 3, 4, 5], dims=("x",), coords={"x": [0, 1, 2, 3, 4]}, name="Entry")
SIMPLE_DATA_SET = xr.Dataset({"Entry": SIMPLE_DATA_ARRAY})


def test_NumpyConverterTuple():
    """
    Checks NumpyConverter using Tuples
    """
    dataset_tuple = (SIMPLE_DATA_SET.copy(), SIMPLE_DATA_SET.copy())
    nc = converter.NumpyConverter()
    np_array = nc.convert_from_xarray(dataset_tuple)
    xr_ds = nc.convert_to_xarray(np_array)

    assert isinstance(xr_ds, tuple)
    assert isinstance(xr_ds[0], xr.Dataset)
    assert isinstance(xr_ds[-1], xr.Dataset)

    xr_ds1 = xr_ds[0]
    assert "Entry" in xr_ds1.data_vars
    xr.testing.assert_identical(xr_ds1["Entry"], SIMPLE_DATA_ARRAY)


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

    # Data that hasn't been converted yet throws runtime error
    with pytest.raises(RuntimeError):
        nc = converter.NumpyConverter()
        _xr_ds = nc.convert_to_xarray(np.array([0, 1, 2]))

    # Data with an empty dataset throws runtime error
    with pytest.raises(RuntimeError):
        nc = converter.NumpyConverter()
        _np_array = nc.convert_from_xarray(SIMPLE_DATA_SET)
        _xr_ds = nc.convert_to_xarray(np.empty(0))

    # String data type throws error
    with pytest.raises(TypeError):
        nc = converter.NumpyConverter()
        _xr_ds = nc.convert_from_xarray(["wrong type"])

    nc = converter.NumpyConverter()

    # Create a DataArray with a 'time' dimension but no 'x' coordinate variable
    data_array = xr.DataArray(np.random.rand(5, 3), dims=["time", "x"], coords={"time": np.random.rand(5)})

    # Create a Dataset from this DataArray
    ds = xr.Dataset({"my_variable": data_array})
    np_array = nc.convert_from_xarray(ds)

    # Value error converting back to numpy array due to missing coord
    with pytest.raises(ValueError):
        _xr_ds = nc.convert_to_xarray(np_array)

    ds = xr.Dataset(
        data_vars={"data_var": ("x", np.array([1, 2, 3]))},
        coords={
            "x": np.arange(3),
            "empty_coord_dim": np.arange(2),  # Coordinate with new dim but no data var
        },
    )
    nc = converter.NumpyConverter()

    # Throw runtime error that cannot record coordinate
    with pytest.raises(RuntimeError):
        _xr_ds = nc.convert_from_xarray(ds)


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

    dataset_tuple = (SIMPLE_DATA_SET.copy(), SIMPLE_DATA_SET.copy())
    da_array = dc.convert_from_xarray(dataset_tuple)

    xr_ds = dc.convert_to_xarray(da_array)
    assert isinstance(xr_ds, tuple)
    assert "Entry" in xr_ds
    xr.testing.assert_identical(xr_ds[0]["Entry"], SIMPLE_DATA_ARRAY)


def test_save_and_load_records(tmpdir):
    """
    Test save and load records functionality.
    """
    tmp_path = tmpdir.mkdir("sub").join("nc.records")

    time_index = pd.date_range("2025-01-01", periods=3, freq="D")

    data = np.random.rand(3, 3, 3)

    coords = {"time": time_index, "x": [np.nan, np.nan, np.nan], "y": np.random.randn(3)}

    test_data_array = xr.DataArray(
        data,
        coords=coords,
        dims=["time", "x", "y"],
    )

    test_dataset = xr.Dataset({"entry": test_data_array})

    nc = converter.NumpyConverter()
    nc.convert_from_xarray(test_data_array)
    nc.save_records(tmp_path)
    saved_records = nc.records.copy()

    # Add extra record to the numpy converter
    nc.convert_from_xarray(test_data_array)

    assert len(saved_records) == 1
    assert len(nc.records) == 2
    assert nc.records != saved_records  # Assert saved records are not the same as converted records

    assert nc.load_records(tmp_path)  # Assert it loads correctly
    assert len(nc.records) == 1

    loaded_records = nc.records.copy()
    loaded_vars = loaded_records[0]["coords"]["x"]
    saved_vars = saved_records[0]["coords"]["x"]

    # Check if variables are equal
    np.testing.assert_equal(saved_vars, loaded_vars)

    # Broken path returns False
    assert not nc.load_records("/broken/path")

    # Trigger datetime instance in save_records.parse
    nc = converter.NumpyConverter()
    nc.convert_from_xarray(test_data_array)
    nc.save_records(tmp_path)

    # Trigger np.isnan(v).all() in save_records.parse
    nc = converter.NumpyConverter()
    da_array = nc.convert_from_xarray(test_dataset)
    nc.convert_to_xarray(da_array)
    nc.save_records(tmp_path)


def test_non_shared_coordinates_throws_value_error():
    """
    Test if coordinates are not shared will throw an appropriate value error
    """
    x_coords = np.array([0, 1, 2])
    y_coords = np.array([0, 1, 2])

    data_1 = np.random.rand(len(x_coords), len(y_coords))
    data_2 = np.random.rand(len(x_coords))
    missing_coord_ds = xr.Dataset(
        {"data_1": (("x", "y"), data_1), "data_2": (("x"), data_2)},
        coords={"x": x_coords, "y": y_coords},
    )
    nc = converter.NumpyConverter()

    # Cannot stack variables so will raise a value error
    with pytest.raises(ValueError):
        nc.convert_from_xarray(missing_coord_ds)
