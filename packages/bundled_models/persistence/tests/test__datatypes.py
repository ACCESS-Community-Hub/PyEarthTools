"""
This test suite tests the use of PetDataset to create a common datatype construction for numpy and
xarray (dataarrays and datasets).

NOTES:
- Since numpy and xarray dataarrays cannot be completely representable by datasets, they will either
  be given dummy variables and dimension names, or user-specified variable and dimension names.
  Creating a common interface to handle all this is tricky.
- While these dummy names are always options when creating a PetDataset, they should not affect
  higher types - e.g. datasets will never be overwritten with the _dummyvarname or "dims()" (because
  it may have several variables wtih different dimensions).
"""

import xarray as xr
import numpy as np
import persistence as pet_persist


def _dummy_sum_fn(x: xr.DataArray, y: int, z: int = 5) -> xr.DataArray:
    """
    Dummy function to test mapping, should return a data array, first argument must be a data array.
    Can take other arguments that may be required for the computation
    """
    return x.sum() + y - z


def test_petdataset_type_homomorphism_numpy():
    """
    Test type mapping with numpy arrays
    """
    # defaults
    test_data = np.ones((5, 2, 3))
    pet_ds = pet_persist.PetDataset(test_data)
    res_ds = pet_ds.map_each_var(_dummy_sum_fn, 5)
    assert "_dummyvarname" in pet_ds.ds.data_vars
    # y = 5
    # z = 5 (default)
    # sum = 5 * 2 * 3 = 30
    assert res_ds["_dummyvarname"] == 30

    # with dummy array naming
    pet_ds = pet_persist.PetDataset(test_data, dummy_varname="new_dummy_name")
    res_ds = pet_ds.map_each_var(_dummy_sum_fn, 5, z=2)
    assert "new_dummy_name" in pet_ds.ds.data_vars
    # y = 5
    # z = 2
    # sum = 5 * 2 * 3 = 30
    # res = sum + 5 - 2 = 33
    assert res_ds["new_dummy_name"] == 33

    # with dimension naming
    pet_ds = pet_persist.PetDataset(test_data, dimnames=["x", "time", "y"])
    res_ds = pet_ds.map_each_var(_dummy_sum_fn, y=-10, z=-15)
    # y = 5
    # z = 2
    # sum = 5 * 2 * 3 = 30
    # res = sum - 10 - (-15) = 35
    assert res_ds["_dummyvarname"] == 35
    assert set(pet_ds.ds.dims) == set(["x", "time", "y"])


def test_petdataset_type_homomorphism_da():
    """
    Test type mapping with data arrays
    """
    # defaults
    test_data = xr.DataArray(np.ones((5, 2, 3)), dims=["the", "last", "resort"])
    pet_ds = pet_persist.PetDataset(test_data)
    res_ds = pet_ds.map_each_var(_dummy_sum_fn, 5)
    assert "_dummyvarname" in pet_ds.ds.data_vars
    # y = 5
    # z = 5 (default)
    # sum = 5 * 2 * 3 = 30
    assert res_ds["_dummyvarname"] == 30

    # with dummy array naming
    pet_ds = pet_persist.PetDataset(test_data, dummy_varname="new_dummy_name")
    res_ds = pet_ds.map_each_var(_dummy_sum_fn, 5, z=2)
    assert "new_dummy_name" in pet_ds.ds.data_vars
    # y = 5
    # z = 2
    # sum = 5 * 2 * 3 = 30
    # res = sum + 5 - 2 = 33
    assert res_ds["new_dummy_name"] == 33

    # with dimension naming
    pet_ds = pet_persist.PetDataset(test_data, dimnames=["x", "time", "y"])
    res_ds = pet_ds.map_each_var(_dummy_sum_fn, y=-10, z=-15)
    # y = 5
    # z = 2
    # sum = 5 * 2 * 3 = 30
    # res = sum - 10 - (-15) = 35
    assert res_ds["_dummyvarname"] == 35
    # dimnames should have no effect on dataarrays
    assert set(pet_ds.ds.dims) == set(["the", "last", "resort"])


def test_petdataset_type_homomorphism_ds():
    """
    Test type mapping with datasets
    """
    # defaults
    test_data = xr.Dataset(
        {
            "potato": xr.DataArray(
                np.ones((5, 2, 3)),
                dims=["the", "last", "resort"],
            ),
            "tomato": xr.DataArray(
                np.ones((2, 1, 2)),
                dims=["x", "y", "z"],
            ),
        }
    )
    pet_ds = pet_persist.PetDataset(test_data)
    res_ds = pet_ds.map_each_var(_dummy_sum_fn, 5)

    # _dummyvarname should be ignored for datasets by default
    assert "_dummyvarname" not in pet_ds.ds.data_vars
    assert res_ds["potato"] == 30
    assert res_ds["tomato"] == 4

    # with dummy array naming
    pet_ds = pet_persist.PetDataset(test_data, dummy_varname="new_dummy_name")
    res_ds = pet_ds.map_each_var(_dummy_sum_fn, 5, z=2)

    # _dummyvarname should be ignored for datasets even when forced
    assert "new_dummy_name" not in pet_ds.ds.data_vars
    assert res_ds["potato"] == 33
    assert res_ds["tomato"] == 7

    # with dimension naming
    pet_ds = pet_persist.PetDataset(test_data, dimnames=["x", "time", "y"])
    res_ds = pet_ds.map_each_var(_dummy_sum_fn, y=-10, z=-15)
    assert res_ds["potato"] == 35
    assert res_ds["tomato"] == 9

    # dimnames should have no effect on dataarrays within the dataset
    assert set(pet_ds.ds["potato"].dims) == set(["the", "last", "resort"])
    assert set(pet_ds.ds["tomato"].dims) == set(["x", "y", "z"])
