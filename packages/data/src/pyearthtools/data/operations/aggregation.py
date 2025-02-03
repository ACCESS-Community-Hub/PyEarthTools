# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
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


from __future__ import annotations

from typing import Callable
import xarray as xr

from pyearthtools.data.transforms import aggregation as aggr_trans


def aggregation(
    dataset: xr.Dataset,
    aggregation: str | Callable,
    reduce_dims: list | str | None = None,
    *,
    preserve_dims: list | str | None = None,
) -> xr.Dataset:
    """Run an aggregation method over a given dataset

    !!! Warning
        Either `reduce_dims` or `preserve_dims` must be given, but not both.

    Args:
        dataset (xr.Dataset):
            Dataset to run aggregation over
        aggregation (str | Callable):
            Aggregation method, can be defined function or xarray function
        reduce_dims (list | str, optional):
            Dimensions to reduce over. Defaults to None.
        preserve_dims (list | str, optional):
            Dimensions to keep. Defaults to None.

    Raises:
        ValueError:
            If invalid `reduce_dims` or `preserve_dims` are given


    Returns:
        (xr.Dataset):
            Dataset with aggregation method applied
    """
    if not reduce_dims and not preserve_dims:
        raise ValueError(f"Either 'reduce_dims' or 'preserve_dims' must be given ")

    if reduce_dims and preserve_dims:
        raise ValueError(f"Both 'reduce_dims' and 'preserve_dims' cannot be given ")

    if reduce_dims:
        aggregation_func = aggr_trans.over(aggregation, reduce_dims)  # type: ignore
    elif preserve_dims:
        aggregation_func = aggr_trans.leaving(aggregation, preserve_dims)  # type: ignore

    return aggregation_func(dataset)
