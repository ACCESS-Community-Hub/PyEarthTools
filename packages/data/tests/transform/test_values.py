import xarray as xr
from pyearthtools.data.transforms.values import SetMissingToNaN
import numpy as np


def test_set_missing_to_nan():
    data = xr.Dataset(
        {
            "total_cloud_cover": ("time", [0, -999, 50]),
            "low_cloud_cover": ("time", [10, -999, 20]),
        }
    )

    varname_val_map = {
        "total_cloud_cover": -999.0,
        "low_cloud_cover": -999.0,
    }

    transform = SetMissingToNaN(varname_val_map)
    transformed_data = transform.apply(data)

    assert np.isnan(transformed_data["total_cloud_cover"].data[1])
    assert np.isnan(transformed_data["low_cloud_cover"].data[1])
    assert transformed_data["total_cloud_cover"].data[2] == 50
