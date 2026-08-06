"""
This suite tests the simple imputer
"""

import persistence as pet_persist
import numpy as np


def test_temporal_imputation_no_missing():
    """
    Nothing should change if there's no missing value
    """
    arr_no_missing = np.full((5, 4, 3), 1, dtype=np.float64)
    imputer = pet_persist.SimpleImpute(arr_no_missing)
    arr_ret = imputer.impute_mean()
    assert np.allclose(arr_ret, arr_no_missing, equal_nan=True)


def test_temporal_imputation_some_missing():
    """
    if some missing, then the nanmean is used to impute.
    """
    # have no missing array for reference
    arr_no_missing = np.full((5, 4, 3), 1, dtype=np.float64)
    # put some nans in a random slab
    arr_some_missing = np.full((5, 4, 3), 1, dtype=np.float64)
    arr_some_missing[1:3, 0:3, 0] = np.nan
    imputer = pet_persist.SimpleImpute(arr_some_missing)
    arr_ret = imputer.impute_mean()
    assert np.allclose(arr_ret, arr_no_missing, equal_nan=True)
    assert np.sum(arr_ret) == 5 * 4 * 3  # (all ones)


def test_temporal_imputation_all_nans():
    """
    If all nan => don't alter original array.
    """
    arr_all_missing = np.full((5, 4, 3), np.nan, dtype=np.float64)
    imputer = pet_persist.SimpleImpute(arr_all_missing)
    arr_ret = imputer.impute_mean()
    assert np.allclose(arr_ret, arr_all_missing, equal_nan=True)
