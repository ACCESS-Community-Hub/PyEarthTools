# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

from __future__ import annotations

from typing import Callable, Union

import xarray as xr

from edit.data.transform import Transform, aggregation
from edit.utils.imports import dynamic_import

known_methods = ["mean", "max", "min", "sum", "std"]


class AggregateTransform:
    """
    Aggregation Transforms,

    | Name | Purpose |
    | ---- | ------- |
    | over | Aggregate over dimensions |
    | leaving | Aggregate Leaving Dimensions |

    """

    @classmethod
    def _get_method(cls, method: Callable | str):
        """
        Check if provided method is valid

        Can be Callable or a known method.

        Args:
            method (Callable | str): Method for aggregation

        Raises:
            AttributeError: If method is invalid
        """
        if method == None or isinstance(method, Callable) or method in known_methods or hasattr(aggregation, method):
            return
        else:
            try:
                dynamic_import(method)
                return
            except (KeyError, ModuleNotFoundError):
                pass
        raise AttributeError(f"{method!r} not recognised nor found to be imported.")

    @classmethod
    def apply(
        cls,
        dataset: xr.Dataset,
        method: Callable | str | dict,
        dimension: str | list[str],
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
        if method == None:
            return dataset
        elif method in known_methods:
            return getattr(dataset, method)(dim=dimension, keep_attrs=True, **kwargs)
        elif hasattr(aggregation, method):
            return getattr(aggregation, method)(dataset, dim=dimension, **kwargs)
        elif isinstance(method, Callable):
            return method(dataset, dimension, **kwargs)
        elif isinstance(method, dict):
            for var in dataset:
                if var not in method and "default" not in method:
                    raise KeyError(f"{var} not in method, and no 'default' is given.")
                dataset[var] = cls.apply(dataset[var], method[var], dimension=dimension, **kwargs)
            return dataset
        else:
            return dynamic_import(method)(dataset, dimension, **kwargs)

    @classmethod
    def over(cls, method: Callable | str | dict, dimension: str | list[str]) -> Transform:
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

        if not isinstance(dimension, (tuple, list)) and dimension is not None:
            dimension = [dimension]
        cls._get_method(method)

        class AggregationTransform(Transform):
            """Aggregate over Given Dimensions"""

            @property
            def _info_(self):
                return dict(dimension=dimension, method=method)

            def apply(self, dataset: xr.Dataset, **kwargs) -> xr.Dataset:
                return cls.apply(dataset, method, dimension, **kwargs)

        return AggregationTransform()

    @classmethod
    def leaving(cls, method: Callable | str | dict, dimension: str | list[str]) -> Transform:
        """
        Get Aggregation Transform to run aggregation method leaving only given dimensions

        Args:
            method (Callable | str | dict): Method to use, can be known method or user defined
            dimension (str | list[str]): Dimensions to leave after aggregation

        Returns:
            (Transform): Transform to apply aggregation
        """

        if not isinstance(dimension, (tuple, list)):
            dimension = [dimension]
        cls._get_method(method)

        class AggregationTransform(Transform):
            """Aggregate Leaving Given Dimensions"""

            @property
            def _info_(self):
                return dict(dimension=dimension, method=method)

            def apply(self, dataset: xr.Dataset) -> xr.Dataset:
                aggregate_dimensions = [elem for elem in dataset.dims if elem not in dimension]
                return cls.apply(dataset, method, aggregate_dimensions)

        return AggregationTransform()
