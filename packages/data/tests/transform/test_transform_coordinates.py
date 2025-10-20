from pyearthtools.data.transforms import coordinates
import xarray as xr
import pytest


# Test data, re-used across tests
lon180 = list(range(-180, 180))
lon360 = list(range(0, 360))
lon_unclear = list(range(10, 120))
data = list(range(-180, 180))
data_unclear = list(range(10, 120))
da180 = xr.DataArray(coords={"longitude": lon180}, dims=["longitude"])
da360 = xr.DataArray(coords={"longitude": lon360}, dims=["longitude"])
da_wrongname = xr.DataArray(coords={"longname": lon180}, dims=["longname"])
da_unclear = xr.DataArray(coords={"longitude": lon_unclear}, dims=["longitude"])


@pytest.fixture()
def ds_vertical():
    return xr.Dataset(
        coords={"longitude": list(range(0, 4)), "vertical": list(range(0, 3))},
        data_vars={"temperature": (["longitude", "vertical"], [[1, 2, 3] for _ in range(4)])},
    )


def test_get_longitude():

    longitude_type = coordinates.get_longitude(da180, transform=False)
    assert longitude_type == "-180-180"
    transform = coordinates.get_longitude(da180, transform=True)
    assert transform._type == "-180-180"

    longitude_type = coordinates.get_longitude(da360, transform=False)
    assert longitude_type == "0-360"
    transform = coordinates.get_longitude(da360, transform=True)
    assert transform._type == "0-360"

    with pytest.raises(ValueError):
        longitude_type = coordinates.get_longitude(da_wrongname, transform=False)

    with pytest.raises(ValueError):
        _result = coordinates.get_longitude(da_unclear, transform=False)


def test_StandardLongitude():

    conform = coordinates.StandardLongitude("0-360")
    fixed = conform.apply(da180)
    assert fixed is not None
    _unchanged = conform.apply(da360)
    assert xr.testing.assert_equal(_unchanged, da360) is None
    assert xr.testing.assert_equal(fixed, da360) is None  # assert_equal returns None if they are equal, not True

    conform = coordinates.StandardLongitude("-180-180")
    fixed = conform.apply(da360)
    _unchanged = conform.apply(da180)
    assert xr.testing.assert_equal(_unchanged, da180) is None
    assert fixed is not None
    assert xr.testing.assert_equal(fixed, da180) is None  # assert_equal returns None if they are equal, not True


def test_ReIndex():

    tf_reindex = coordinates.ReIndex({"longitude": "reversed"})
    _reversed = tf_reindex.apply(da180)
    expected_result = xr.DataArray(coords={"longitude": lon180[::-1]}, dims=["longitude"])
    assert xr.testing.assert_equal(_reversed, expected_result) is None


def test_Select():

    tf_select = coordinates.Select({"longitude": slice(10, 120)})
    result = tf_select.apply(da180)
    assert result is not None
    expected_result = xr.DataArray(coords={"longitude": list(range(10, 121))}, dims=["longitude"])
    assert xr.testing.assert_equal(result, expected_result) is None


def test_Drop(ds_vertical):

    tf_drop = coordinates.Drop("vertical")
    _result = tf_drop.apply(ds_vertical)
    assert "vertical" not in _result.variables


def test_Assign(ds_vertical):

    tf_assign = coordinates.Assign({"longitude": list(range(0, 4)), "vertical": list(range(3, 6))})

    _result = tf_assign.apply(ds_vertical)
    assert set(_result.coords.keys()) == {"longitude", "vertical"}
    assert _result.longitude.values.tolist() == list(range(0, 4))
    assert _result.vertical.values.tolist() == list(range(3, 6))


# TODO: possible bug - the padding values of data_vars (temperature) are NaN, they should be set to some fill value
# def test_Pad(ds_vertical):
#     tf_pad = coordinates.Pad({"longitude": (1, 2)})
#
#     _result = tf_pad.apply(ds_vertical)
#
#     assert set(_result.coords.keys()) == {"longitude", "vertical"}
#     assert _result.dims["longitude"] == ds_vertical.dims["longitude"] + 3
#
#     expected_longitude = [-1, 0, 1, 2, 3, 4, 5]
#     assert _result.longitude.values.tolist() == expected_longitude
#
#     with pytest.raises(AssertionError):
#         assert not np.isnan(_result.sel(longitude=-1, vertical=1).temperature.values)
