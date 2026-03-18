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

from pyearthtools.pipeline.operations.xarray.join import Merge

import numpy as np
import xarray as xr


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
