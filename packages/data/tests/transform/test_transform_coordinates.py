from pyearthtools.data.transforms import coordinates
import xarray as xr
import numpy as np
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

ds_vertical = xr.Dataset(
    coords={"longitude": list(range(0, 4)), "vertical": list(range(0, 3))},
    data_vars={"temperature": (["longitude", "vertical"], np.random.rand(4, 3))},
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
    # TODO - shouldn't this be true?
    # assert xr.testing.assert_equal(fixed, da360)

    conform = coordinates.StandardLongitude("-180-180")
    fixed = conform.apply(da360)
    _unchanged = conform.apply(da180)
    assert fixed is not None
    # TODO - shouldn't this be true?
    # assert xr.testing.assert_equal(fixed, da180)


def test_ReIndex():

    tf_reindex = coordinates.ReIndex({"longitude": "reversed"})
    tf_reindex.apply(da180)
    # TODO: Assert the range of the reversed coordinate is 180 to -180


def test_Select():

    tf_select = coordinates.Select({"longitude": slice(10, 120)})
    result = tf_select.apply(da180)
    assert result is not None
    # TODO: Check the result against the requested slice


def test_Drop():

    tf_drop = coordinates.Drop("vertical")
    _result = tf_drop.apply(ds_vertical)

    # TODO: Assert that the dimension has been dropped


def test_Flatten():

    tf_flatten = coordinates.Flatten("vertical")
    _result = tf_flatten.apply(ds_vertical)

    # TODO: Assert the flattened data looks correct


def test_Expand():

    tf_expand = coordinates.Expand("vertical")
    _result = tf_expand.apply(ds_vertical)

    # TODO: Assert the expanded dataset has a vertical dimension


def test_SelectFlatten():

    tf_selectflatten = coordinates.SelectFlatten({"longitude": slice(10, 120)})

    with pytest.raises(NotImplementedError):
        _result = tf_selectflatten.apply(ds_vertical)
        # TODO fix the code or avoid the issue

    # TODO: Check the values of the resulting dataset


def test_Assign():

    tf_assign = coordinates.Assign({"longitude": list(range(0, 4)), "vertical": list(range(3, 6))})

    _result = tf_assign.apply(ds_vertical)

    # TODO: check the values of the vertical coords


def test_Pad():
    tf_pad = coordinates.Pad({"longitude": list(range(0, 4))})

    with pytest.raises(ValueError):
        _result = tf_pad.apply(ds_vertical)
        # TODO: Fix the code or fix the test

    # TODO: check the values of the result


def test_weak_cast_to_int():

    wcti = coordinates.weak_cast_to_int

    assert wcti(5.0) == 5
    assert isinstance(wcti(5.0), int)

    assert wcti("hello") == "hello"
