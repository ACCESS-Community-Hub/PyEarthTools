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

from typing import Callable, Optional, Union

import xarray as xr

from pyearthtools.data.transforms import Transform, aggregation
from pyearthtools.utils.initialisation.imports import dynamic_import

known_methods = ["mean", "max", "min", "sum", "std"]


class Aggregate(Transform):
    """
    Aggregation Transforms,
    """

    def __init__(
        self,
        method: Callable | str | dict[str, Callable | str],
        reduce_dims: Optional[list[str] | str] = None,
        keep_dims: Optional[list[str] | str] = None,
    ):
        super().__init__()
        self.record_initialisation()

        if reduce_dims and keep_dims:
            raise ValueError("Cannot provide both `reduce_dims` and `keep_dims`.")

        if not isinstance(reduce_dims, (tuple, list)) and reduce_dims is not None:
            reduce_dims = [reduce_dims]
        if not isinstance(keep_dims, (tuple, list)) and keep_dims is not None:
            keep_dims = [keep_dims]

        self.reduce_dims = reduce_dims
        self.keep_dims = keep_dims
        self.method = method

    @classmethod
    def _get_method(cls, method: Callable | str | dict[str, Callable | str]):
        """
        Check if provided method is valid

        Can be Callable or a known method.

        Args:
            method (Callable | str): Method for aggregation

        Raises:
            AttributeError: If method is invalid
        """
        if (
            method == None
            or isinstance(method, (dict, Callable))
            or method in known_methods
            or hasattr(aggregation, method)
        ):
            return
        else:
            try:
                dynamic_import(method)
                return
            except (KeyError, ModuleNotFoundError):
                pass
        raise AttributeError(f"{method!r} not recognised nor found to be imported.")

    def apply(
        self,
        dataset: xr.Dataset,
        **kwargs,
    ) -> xr.Dataset:
        """
        Apply Aggregation to Dataset

        Args:
            dataset (xr.Dataset): Dataset to apply aggregation to
            method (Callable | str): Method of aggregation, either func or string
            dimension (str | list[str]): Dimension to apply aggregation on

        Returns:
            (xr.Dataset): Aggregated Dataset
        """
        method = self.method

        if self.keep_dims:
            dimension = [elem for elem in dataset.dims if elem not in self.keep_dims]
        else:
            dimension = self.reduce_dims

        self._get_method(method)

        if method is None:
            return dataset

        elif method in known_methods:
            return getattr(dataset, method)(dim=dimension, keep_attrs=True, **kwargs)

        elif isinstance(method, str) and hasattr(aggregation, method):
            return getattr(aggregation, method)(dataset, dim=dimension, **kwargs)

        elif isinstance(method, Callable):
            return method(dataset, dimension, **kwargs)

        elif isinstance(method, dict):
            for var in dataset:
                if var not in method and "default" not in method:
                    raise KeyError(f"{var} not in method, and no 'default' is given.")

                dataset[var] = Aggregate(method[str(var)], reduce_dims=str(dimension))(dataset[[var]], **kwargs)
            return dataset
        else:
            return dynamic_import(method)(dataset, dimension, **kwargs)  # type: ignore


def over(dimension: str | list[str], method: Callable | str | dict) -> Aggregate:
    """
    Get Aggregation Transform to run aggregation method over given dimensions

    Args:
        method (Callable | str | dict):
            Method to use, can be known method or user defined
        dimension (str | list[str]):
            Dimensions to run aggregation over

    Returns:
        (Transform):
            Transform to apply aggregation
    """

    return Aggregate(reduce_dims=dimension, method=method)


def leaving(method: Callable | str | dict, dimension: str | list[str]) -> Transform:
    """
    Get Aggregation Transform to run aggregation method leaving only given dimensions

    Args:
        method (Callable | str | dict): Method to use, can be known method or user defined
        dimension (str | list[str]): Dimensions to leave after aggregation

    Returns:
        (Transform): Transform to apply aggregation
    """

    return Aggregate(keep_dims=dimension, method=method)
