# Copyright Commonwealth of Australia, Bureau of Meteorology 2026.
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

from pyearthtools.pipeline.operations.xarray.join import (
    Merge,
    LatLonInterpolate,
    GeospatialTimeSeriesMerge,
    InterpLike,
    Concatenate,
)

import numpy as np
import xarray as xr

import pytest


def test_merge():
    coords = {"x": [1, 2, 3], "y": [4, 5, 6]}

    da = xr.DataArray(
        np.arange(9).reshape(3, 3),
        coords=coords,
        dims=["x", "y"],
        name="alpha",
        attrs={"source": "model", "units": "K"},
    )

    ds = xr.Dataset(
        {
            "beta": xr.DataArray(np.arange(9, 18).reshape(3, 3), coords=coords, dims=["x", "y"]),
            "gamma": xr.DataArray(np.arange(18, 27).reshape(3, 3), coords=coords, dims=["x", "y"]),
        },
        attrs={"source": "model", "resolution": "1deg"},  # "source" overlaps with da
    )

    sample = (da, ds)

    joiner = Merge()

    result = joiner.join(sample)

    assert result["alpha"].equals(da), "Merge.join didn't merge objects correctly."
    assert result["beta"].equals(ds["beta"]), "Merge.join didn't merge objects correctly."
    assert result["gamma"].equals(ds["gamma"]), "Merge.join didn't merge objects correctly."
    assert result.attrs == da.attrs, "Merge.join result didn't preserve first object's attributes"
    assert result.attrs != ds.attrs, "Merge.join didn't discard second object's attributes"

    unjoined = joiner.unjoin(result)

    assert isinstance(unjoined, tuple), "Merge.unjoin didn't result in a tuple."
    for d_undo, d_orig in zip(unjoined, sample, strict=True):
        assert isinstance(d_undo, type(d_orig))
        assert d_undo.equals(d_orig), "Merge.unjoin didn't restore objects."
        assert d_undo.attrs == d_orig.attrs, "Merge.unjoin didn't preserve attributes."

    # test passing kwargs to xr.merge
    # should combine attributes
    joiner = Merge(merge_kwargs={"combine_attrs": "no_conflicts"})

    result = joiner.join(sample)

    assert result["alpha"].equals(da), 'passing combine_attrs="no_conflict" to Merge didn\'t merge object correctly.'
    assert result["beta"].equals(
        ds["beta"]
    ), 'passing combine_attrs="no_conflict" to Merge didn\'t merge object correctly.'
    assert result["gamma"].equals(
        ds["gamma"]
    ), 'passing combine_attrs="no_conflict" to Merge didn\'t merge object correctly.'
    assert result.attrs == (
        da.attrs | ds.attrs
    ), 'passing combine_attrs="no_conflict" to Merge didn\'t unionise attributes.'

    unjoined = joiner.unjoin(result)

    assert isinstance(
        unjoined, tuple
    ), 'passing combine_attrs="no_conflict" to Merge didn\'t result in a tuple when unjoining.'
    for d_undo, d_orig in zip(unjoined, sample, strict=True):
        assert isinstance(
            d_undo, type(d_orig)
        ), "passing combine_attrs=\"no_conflict\" to Merge didn't preserve object's type when unjoining."
        assert d_undo.equals(
            d_orig
        ), 'passing combine_attrs="no_conflict" to Merge didn\'t restore object when unjoining.'
        assert (
            d_undo.attrs == d_orig.attrs
        ), 'passing combine_attrs="no_conflict" to Merge didn\'t preserve attributes when unjoining.'


def _make_ds(var_name, data, lat, lon, time=None, lat_name="latitude", lon_name="longitude"):
    """Create a Dataset with latitude, longitude, and time coords."""
    time = time or [0]
    return xr.Dataset(
        {var_name: xr.DataArray(data, dims=["time", lat_name, lon_name])},
        coords={"time": time, lat_name: lat, lon_name: lon},
    )


@pytest.fixture
def ds_ref():
    return _make_ds(var_name="var_ref", data=np.arange(9).reshape(1, 3, 3), lat=[0.0, 1.0, 2.0], lon=[0.0, 1.0, 2.0])


@pytest.mark.parametrize(
    ("lat_name", "lon_name", "joiner_factory"),
    [
        pytest.param(
            "latitude", "longitude", lambda ds_ref: LatLonInterpolate(reference_index=0), id="reference_index"
        ),
        pytest.param("lat", "lon", lambda ds_ref: LatLonInterpolate(reference_dataset=ds_ref), id="reference_dataset"),
    ],
)
def test_latlon_interpolate_join(lat_name, lon_name, joiner_factory):
    """Tests that LatLonInterpolate merges and interpolates datasets to the reference grid."""
    ds_ref = _make_ds(
        var_name="var1",
        data=np.arange(9).reshape(1, 3, 3),
        lat=[0.0, 1.0, 2.0],
        lon=[0.0, 1.0, 2.0],
        lat_name=lat_name,
        lon_name=lon_name,
    )
    ds_coarse = _make_ds(
        var_name="var2",
        data=np.arange(9, 13).reshape(1, 2, 2),
        lat=[-0.25, 2.25],
        lon=[-0.25, 2.25],
        lat_name=lat_name,
        lon_name=lon_name,
    )

    result = joiner_factory(ds_ref).join((ds_ref, ds_coarse))

    assert "var1" in result.data_vars
    assert "var2" in result.data_vars
    # astype is needed because interp changes datatype
    assert ds_ref["var1"].equals(result["var1"].astype(int))
    assert ds_ref.coords.equals(result["var2"].coords)

    expected_interp = np.array([[9.0, 9.0, 10.0], [9.0, 9.0, 10.0], [11.0, 11.0, 12.0]])
    assert np.array_equal(result["var2"].squeeze("time").values, expected_interp)


def test_latlon_interpolate_errors(ds_ref):
    """Tests that LatLonInterpolate raises errors for invalid configurations."""
    ds_coarse = _make_ds(var_name="var2", data=np.arange(9, 13).reshape(1, 2, 2), lat=[-0.25, 2.25], lon=[-0.25, 2.25])

    with pytest.raises(ValueError):
        LatLonInterpolate()

    with pytest.raises(ValueError):
        LatLonInterpolate(reference_dataset=ds_ref, reference_index=0)

    with pytest.raises(ValueError):
        LatLonInterpolate(reference_dataset=ds_ref.rename({"latitude": "abc", "longitude": "123"}))

    joiner = LatLonInterpolate(reference_index=0)
    joiner.reference_index = None
    with pytest.raises(ValueError):
        joiner.join((ds_ref, ds_coarse))

    # unjoin not implemented
    with pytest.raises(NotImplementedError):
        joiner.unjoin(ds_ref)


def test_geospatial_timeseries_merge_join(ds_ref):
    """Tests that GeospatialTimeSeriesMerge interpolates and merges datasets."""
    da_coarse = xr.DataArray(
        np.arange(9, 13).reshape(1, 2, 2),
        dims=["time", "latitude", "longitude"],
        coords={"time": [0], "latitude": [-0.25, 2.25], "longitude": [-0.25, 2.25]},
        name="var2",
    )

    joiner = GeospatialTimeSeriesMerge(reference_index=0)
    result = joiner.join((ds_ref, da_coarse))

    assert "var_ref" in result.data_vars
    assert "var2" in result.data_vars
    assert (
        result["var2"].shape == ds_ref["var_ref"].shape
    ), "GeospatialTimeSeriesMerge did not interpolate to reference grid shape."
    assert tuple(result.latitude.values) == tuple(ds_ref.latitude.values)
    assert tuple(result.longitude.values) == tuple(ds_ref.longitude.values)


def test_geospatial_timeseries_merge_errors(ds_ref):
    """Tests that GeospatialTimeSeriesMerge raises errors for invalid inputs."""
    ds_no_time = _make_ds(
        var_name="var2", data=np.arange(9, 18).reshape(1, 3, 3), lat=ds_ref.latitude.values, lon=ds_ref.longitude.values
    ).drop_dims("time")

    # fail when trying to join without setting reference
    with pytest.raises(ValueError):
        GeospatialTimeSeriesMerge().join((ds_ref, ds_ref))

    joiner = GeospatialTimeSeriesMerge(reference_dataset=ds_ref)
    # fail when trying to join datasets and one doesn't have the time dim
    with pytest.raises(ValueError):
        joiner.join((ds_no_time, ds_ref))
    with pytest.raises(ValueError):
        joiner.join((ds_ref, ds_no_time))

    # fail when trying to unjoin
    with pytest.raises(NotImplementedError):
        GeospatialTimeSeriesMerge().unjoin(None)


def test_interplike(ds_ref):

    da_coarse = xr.DataArray(
        np.arange(9, 13).reshape(2, 2),
        dims=["latitude", "longitude"],
        coords={"latitude": [-0.25, 2.25], "longitude": [-0.25, 2.25]},
        name="var1",
    )
    da_fine = xr.DataArray(
        np.arange(13, 29).reshape(4, 4),
        dims=["latitude", "longitude"],
        coords={"latitude": [0.0, 0.67, 1.33, 2.0], "longitude": [0.0, 0.67, 1.33, 2.0]},
        name="var2",
    )

    # test default interpolation method (nearest)
    joiner = InterpLike(reference_dataset=ds_ref)
    result = joiner.join([da_coarse, da_fine])
    expected_nearest = {
        "var1": np.array([[9.0, 9.0, 10.0], [9.0, 9.0, 10.0], [11.0, 11.0, 12.0]]),
        "var2": np.array([[13.0, 14.0, 16.0], [17.0, 18.0, 20.0], [25.0, 26.0, 28.0]]),
    }
    for ds in ("var1", "var2"):
        assert ds in result.data_vars, f"{ds} missing from joined dataset"
        assert (1,) + result[ds].shape == ds_ref[
            "var_ref"
        ].shape, f"InterpLike didn't interpolate {ds} onto ds_ref's coords"
        assert np.array_equal(expected_nearest[ds], result[ds].values), f"Interplike didn't interpolate {ds}'s values"

    # test linear interpolation method
    joiner = InterpLike(reference_dataset=ds_ref, method="linear")
    result = joiner.join([da_coarse, da_fine])
    expected_linear = {
        "var1": np.array([[9.3, 9.7, 10.1], [10.1, 10.5, 10.9], [10.9, 11.3, 11.7]]),
        "var2": np.array([[13.0, 14.5, 16.0], [19.0, 20.5, 22.0], [25.0, 26.5, 28.0]]),
    }
    for ds in ("var1", "var2"):
        assert np.allclose(expected_linear[ds], result[ds].values)

    # test reference index
    joiner = InterpLike(reference_index=0)
    result = joiner.join([ds_ref, da_coarse, da_fine])
    assert "var_ref" in result.data_vars, "InterpLike didn't preserve reference dataset"
    assert ds_ref["var_ref"].equals(result["var_ref"].astype(int)), "InterpLike didn't reproduce reference"
    for ds in ("var1", "var2"):
        assert ds in result.data_vars, f"{ds} missing from joined dataset"
        assert (1,) + result[ds].shape == ds_ref[
            "var_ref"
        ].shape, f"InterpLike didn't interpolate {ds} onto ds_ref's coords"
        assert np.array_equal(expected_nearest[ds], result[ds].values), f"Interplike didn't interpolate {ds}'s values"


def test_interplike_errors(ds_ref):
    joiner = InterpLike()
    with pytest.raises(ValueError):
        joiner.join([ds_ref])

    with pytest.raises(NotImplementedError):
        joiner.unjoin(ds_ref)


def test_concatenate():
    # test with dataarrays
    da1 = xr.DataArray(np.arange(6).reshape((2, 3)), coords={"x": range(2), "y": range(3)})
    da2 = xr.DataArray(np.arange(6, 18).reshape((4, 3)), coords={"x": range(4), "y": range(3)})
    joiner = Concatenate(concat_dim="x")
    result = joiner.join([da1, da2])
    assert np.array_equal(result.values, np.arange(18).reshape((6, 3)))

    # test with datasets
    ds1 = xr.Dataset({"var1": da1})
    ds2 = xr.Dataset({"var2": da2})
    result = joiner.join([ds1, ds2])
    expected = np.vstack((da1.values, np.full((4, 3), np.nan)))
    assert np.array_equal(expected, result["var1"].values, equal_nan=True)
    expected = np.vstack((np.full((2, 3), np.nan), da2.values))
    assert np.array_equal(expected, result["var2"].values, equal_nan=True)

    # test concat kwargs (dim kwarg should be ignored)
    joiner = Concatenate(concat_dim="x", concat_kwargs={"fill_value": 0, "dim": "y"})
    result = joiner.join([ds1, ds2])
    expected = np.vstack((da1.values, np.zeros((4, 3))))
    assert np.array_equal(
        expected,
        result["var1"].values,
    )
    expected = np.vstack((np.zeros((2, 3)), da2.values))
    assert np.array_equal(
        expected,
        result["var2"].values,
    )

    # unjoin not implemented: returns the input
    joiner = Concatenate(concat_dim="x")
    assert ds1.equals(joiner.unjoin(ds1))
