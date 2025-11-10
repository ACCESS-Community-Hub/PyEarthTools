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

import pytest
import xarray as xr

from pyearthtools.pipeline.operations.xarray import _sort as sort, AlignDataVariableDimensionsToDatasetCoords

SIMPLE_DA1 = xr.DataArray(
    [
        [
            [0.9, 0.0, 5],
            [0.7, 1.4, 2.8],
            [0.4, 0.5, 2.3],
        ],
        [
            [1.9, 1.0, 1.5],
            [1.7, 2.4, 1.1],
            [1.4, 1.5, 3.3],
        ],
    ],
    coords=[[10, 20], [0, 1, 2], [5, 6, 7]],
    dims=["height", "lat", "lon"],
)
SIMPLE_DS1 = xr.Dataset({"Temperature": SIMPLE_DA1})
SIMPLE_DS2 = xr.Dataset({"Humidity": SIMPLE_DA1, "Temperature": SIMPLE_DA1, "WombatsPerKm2": SIMPLE_DA1})
SIMPLE_DA_WITH_NAMED_COORDS = xr.DataArray(
    SIMPLE_DA1.data,
    coords={"h": ("height", [10, 20]), "x": ("lat", [0, 1, 2]), "y": ("lon", [5, 6, 7])},
    dims=["height", "lat", "lon"],
)


@pytest.mark.parametrize("undo,unaligned_dataarray", [(True, SIMPLE_DA1), (False, SIMPLE_DA_WITH_NAMED_COORDS)])
def test_align(undo, unaligned_dataarray):
    """Tests that the dataset dimension alignment operation works."""

    # instantiate align op
    align_op = AlignDataVariableDimensionsToDatasetCoords(restore_dim_order_on_undo=undo)

    # create dataset with dims that are unaligned
    unaligned_dataset = xr.Dataset(
        {
            "Temperature": unaligned_dataarray.transpose("lat", "lon", "height"),
            "Humidity": unaligned_dataarray,
            "WombatsPerKm2": unaligned_dataarray.transpose("lon", "height", "lat"),
        }
    )

    # apply aligner to dataset and check that dataset dims now align
    ds_aligned = align_op.apply_func(unaligned_dataset)
    assert (
        ds_aligned["Temperature"].dims == ds_aligned["Humidity"].dims
    ), "Humidity DataArray dims not aligned to Temperature DataArray dims."
    assert (
        ds_aligned["Temperature"].dims == ds_aligned["WombatsPerKm2"].dims
    ), "WombatsPerKm2 DataArray dims not aligned to Temperature DataArray dims."

    # test undo alignment
    ds_aligned_undone = align_op.undo_func(ds_aligned)
    assert (ds_aligned_undone == ds_aligned).all(), "Underlying data was modified in undoing the dimension alignment."
    for array_name in unaligned_dataset:
        if undo:
            assert (
                ds_aligned_undone[array_name].dims == unaligned_dataset[array_name].dims
            ), "Undoing dimensions failed."
        else:
            assert ds_aligned_undone[array_name].dims == ds_aligned[array_name].dims, "Dimensions changed."


def test_Sort():

    s = sort.Sort()
    s.apply_func(SIMPLE_DS1)

    s = sort.Sort(order=["Temperature", "Humidity"], strict=False)
    r = s.apply_func(SIMPLE_DS2)
    assert list(r.data_vars) == ["Temperature", "Humidity", "WombatsPerKm2"]

    s = sort.Sort(order=["Humidity", "Temperature"], strict=False)
    r = s.apply_func(SIMPLE_DS2)
    assert list(r.data_vars) == ["Humidity", "Temperature", "WombatsPerKm2"]

    s = sort.Sort(order=["Temperature", "Humidity", "KangaroosPerCmSq"], strict=False)
    r = s.apply_func(SIMPLE_DS2)
    assert list(r.data_vars) == ["Temperature", "Humidity", "WombatsPerKm2"]

    s = sort.Sort(order=["Temperature", "Humidity", None], strict=False)
    r = s.apply_func(SIMPLE_DS2)
    assert list(r.data_vars) == ["Temperature", "Humidity", "WombatsPerKm2"]

    s = sort.Sort(order=["Temperature", "Humidity", "WombatsPerKm2"], strict=True)
    s.apply_func(SIMPLE_DS2)

    with pytest.raises(RuntimeError):
        s = sort.Sort(order=["Temperature", "Humidity"], strict=True)
        s.apply_func(SIMPLE_DS2)
