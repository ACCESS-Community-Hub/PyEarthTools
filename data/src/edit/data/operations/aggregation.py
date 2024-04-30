# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

from __future__ import annotations

from typing import Callable
import xarray as xr

from edit.data.transform import aggregation as aggr_trans


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
