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


from functools import reduce
from typing import TypeVar, Union, Optional, Any

import xarray as xr

from pyearthtools.pipeline.branching.join import Joiner

T = TypeVar("T", xr.Dataset, xr.DataArray)


class Merge(Joiner):
    """
    Merge a tuple of xarray object's.

    Currently cannot undo this operation
    """

    _override_interface = "Serial"

    def __init__(self, merge_kwargs: Optional[dict[str, Any]] = None):
        super().__init__()
        self.record_initialisation()
        self._merge_kwargs = merge_kwargs

    def join(self, sample: tuple[Union[xr.Dataset, xr.DataArray], ...]) -> xr.Dataset:
        """Join sample"""
        return xr.merge(sample, **(self._merge_kwargs or {}))

    def unjoin(self, sample: Any) -> tuple:
        return super().unjoin(sample)


class GeospatialTimeSeriesMerge(Joiner):
    """
    The default "merge" and "interplike" xarray commands are very general
    This can result in 'merged' or 'interpolated' arrays which don't do
    what's hoped for but don't raise any errors either.

    This joiner is more strict about the merging and interpolating, and also
    raises more informative error messages when it runs into trouble.
    """

    _override_interface = "Serial"

    def __init__(
        self,
        reference_dataset=None,
        reference_index=None,
        interpolation_method="nearest",
        time_dimension="time",
        merge_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.record_initialisation()
        self.reference_dataset = reference_dataset
        self.reference_index = reference_index
        self.interpolation_method = interpolation_method
        self.time_dimension = time_dimension
        self._merge_kwargs = merge_kwargs

    def _join_two_datasets(self, sample_a: xr.Dataset, sample_b: xr.Dataset) -> xr.Dataset:
        """
        Used to reduce a sequence of joinable items. Only called by the public interface join method.
        """

        # Check each sample has the proper time dimension
        if self.time_dimension not in sample_a.coords:
            raise ValueError(f"Time dimension missing from {str(sample_a)}")

        if self.time_dimension not in sample_b.coords:
            raise ValueError(f"Time dimension missing from {str(sample_b)}")

        interped_a = sample_a.interp_like(self.reference_dataset, method="nearest")
        interped_b = sample_b.interp_like(self.reference_dataset, method="nearest")
        merged = xr.merge([interped_a, interped_b])
        return merged

    def join(self, sample: tuple[Union[xr.Dataset, xr.DataArray], ...]) -> xr.Dataset:
        """Join sample"""

        # Obtain the reference dataset
        if self.reference_dataset is None:
            if self.reference_index is not None:
                self.reference_dataset = sample[self.reference_index]
            else:
                raise ValueError("No reference dataset or reference index set")

        merged = reduce(lambda a, b: self._join_two_datasets(a, b), sample)
        return merged

    def unjoin(self, sample: Any) -> tuple:
        raise NotImplementedError("Not Implemented")


class InterpLike(Joiner):
    """
    Merge a tuple of xarray object's.

    Currently cannot undo this operation
    """

    _override_interface = "Serial"

    def __init__(
        self,
        reference_dataset=None,
        reference_index=None,
        method="nearest",
        merge_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.record_initialisation()
        self.reference_dataset = reference_dataset
        self.reference_index = reference_index
        self.interp_method = method
        self._merge_kwargs = merge_kwargs

    def join(self, sample: tuple[Union[xr.Dataset, xr.DataArray], ...]) -> xr.Dataset:
        """Join sample"""
        # merged = reduce(lambda a, b: a.interp_like(b), sample)

        if self.reference_dataset is not None:
            reference = self.reference_dataset
        elif self.reference_index is not None:
            reference = sample[self.reference_index]
        else:
            raise ValueError("No reference dataset or reference index set")

        interped = [i.interp_like(reference, method=self.interp_method) for i in sample]
        merged = xr.merge(interped)
        return merged

    def unjoin(self, sample: Any) -> tuple:
        raise NotImplementedError("Not Implemented")


class Concatenate(Joiner):
    """
    Concatenate a tuple of xarray object's

    Currently cannot undo this operation
    """

    _override_interface = "Serial"

    def __init__(self, concat_dim: str, concat_kwargs: Optional[dict[str, Any]] = None):
        super().__init__()
        self.record_initialisation()
        self._concat_dim = concat_dim

        if concat_kwargs:
            concat_kwargs.pop("dim", None)

        self._concat_kwargs = concat_kwargs

    def join(self, sample: tuple[T, ...]) -> T:
        """Concat sample"""
        return xr.concat(sample, dim=self._concat_dim, **(self._concat_kwargs or {}))  # type: ignore

    def unjoin(self, sample: Any) -> tuple:
        return super().unjoin(sample)
