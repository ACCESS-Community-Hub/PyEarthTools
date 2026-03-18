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

from pyearthtools.pipeline.operations.xarray.join import Merge, LatLonInterpolate

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


def _make_ds(var_name, data, lat, lon, lat_name="latitude", lon_name="longitude"):
    """Create a Dataset with lat/lon coords."""
    return xr.Dataset(
        {var_name: xr.DataArray(data, dims=[lat_name, lon_name])},
        coords={lat_name: lat, lon_name: lon},
    )


EXPECTED_INTERPOLATED = np.array([[9.0, 9.0, 10.0], [9.0, 9.0, 10.0], [11.0, 11.0, 12.0]])


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
    ds_ref = _make_ds("ds1", np.arange(9).reshape(3, 3), [0.0, 1.0, 2.0], [0.0, 1.0, 2.0], lat_name, lon_name)
    ds_coarse = _make_ds("ds2", np.arange(9, 13).reshape(2, 2), [-0.25, 2.25], [-0.25, 2.25], lat_name, lon_name)

    result = joiner_factory(ds_ref).join((ds_ref, ds_coarse))

    assert "ds1" in result.data_vars
    assert "ds2" in result.data_vars
    # astype is needed because interp changes datatype
    assert ds_ref["ds1"].equals(result["ds1"].astype(int))
    assert ds_ref.coords.equals(result["ds2"].coords)
    assert np.array_equal(result["ds2"].values, EXPECTED_INTERPOLATED)


def test_latlon_interpolate_errors():
    """Tests that LatLonInterpolate raises errors for invalid configurations."""
    ds_ref = _make_ds("ds1", np.arange(9).reshape(3, 3), [0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    ds_coarse = _make_ds("ds2", np.arange(9, 13).reshape(2, 2), [-0.25, 2.25], [-0.25, 2.25])

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


def test_latlon_interpolate_unjoin_not_implemented():
    """Tests that LatLonInterpolate.unjoin raises NotImplementedError."""
    ds_ref = _make_ds("ds1", np.arange(9).reshape(3, 3), [0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    joiner = LatLonInterpolate(reference_dataset=ds_ref)
    with pytest.raises(NotImplementedError):
        joiner.unjoin(ds_ref)
