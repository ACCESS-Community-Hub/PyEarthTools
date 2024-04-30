# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

from __future__ import annotations

import functools
import warnings

import numpy as np
import xarray as xr

xr.set_options(keep_attrs=True)

from edit.data.transform.normalisation.default import normaliser, open_file
from edit.data.transform.transform import FunctionTransform, Transform


class Normalise(normaliser):
    """
    Normalise incoming data.

    Either call this class, or get attribute for specific normalisation strategy
    """

    @functools.wraps(normaliser.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, key: str):
        function = self._find_user_normaliser(key).normalise
        if isinstance(function, Transform):
            return function
        return FunctionTransform(function)
    
    @property
    def anomaly(normalise_self):
        """
        Normalise datasets using anomalies.
        """

        class AnomalyNormaliser(Transform):
            """Normalise Dataset by creating anomalies"""

            @property
            def _info_(self):
                return normalise_self._info_

            def apply(self, dataset: xr.Dataset):
                dataset = xr.Dataset(dataset)

                for variable_name in dataset:
                    average = normalise_self.get_anomaly(variable_name)

                    dataset[variable_name] = dataset[variable_name] - average[variable_name]
                return dataset

        return AnomalyNormaliser()

    @property
    def range(normalise_self):
        """
        Normalise datasets betweens 0 and 1 using range.
        """

        class RangeNormaliser(Transform):
            """Normalise between 0-1 with ranges"""

            @property
            def _info_(self):
                return normalise_self._info_

            @property
            def __doc__(self):
                return f"Normalise between 0-1 with ranges, with {normalise_self.index.__class__.__name__} "

            def apply(self, dataset: xr.Dataset):
                dataset = xr.Dataset(dataset)

                for variable_name in dataset:
                    range = normalise_self.get_range(variable_name)

                    dataset[variable_name] = (dataset[variable_name] - range[variable_name]["min"]) / (
                        range[variable_name]["max"] - range[variable_name]["min"]
                    )

                return dataset

        return RangeNormaliser()

    def manual_range(normalise_self, min: float, max: float):
        warnings.warn(f"This function is being deprecated, use `range` method and set override.", DeprecationWarning)

        class ManualRangeNormaliser(Transform):
            """Normalise between 0-1 with ranges"""
            @property
            def _info_(self):
                return normalise_self._info_
            
            @property
            def __doc__(self):
                return f"Normalise between 0-1 with given ranges {min}, {max}"

            def apply(self, dataset: xr.Dataset):
                dataset = xr.Dataset(dataset)
                for variable_name in dataset:
                    dataset[variable_name] = (dataset[variable_name] - min) / (max - min)

                return dataset

        return ManualRangeNormaliser()

    @property
    def deviation(normalise_self):
        """
        Normalise datasets using mean & standard deviation
        """

        class DeviationNormaliser(Transform):
            """Normalise Dataset using mean & standard deviation"""

            @property
            def _info_(self):
                return normalise_self._info_

            def apply(self, dataset: xr.Dataset):
                dataset = xr.Dataset(dataset)
                for variable_name in dataset:
                    average, deviation = normalise_self.get_deviation(variable_name)

                    dataset[variable_name] = (dataset[variable_name] - average[variable_name]) / deviation[
                        variable_name
                    ]

                return dataset

        return DeviationNormaliser()


    @property
    def deviation_spatial(normalise_self):
        """
        Normalise datasets using mean & standard deviation
        """

        class DeviationNormaliser(Transform):
            """Normalise Dataset using mean & standard deviation spatially"""

            @property
            def _info_(self):
                return normalise_self._info_

            def apply(self, dataset: xr.Dataset):
                dataset = xr.Dataset(dataset)
                for variable_name in dataset:
                    average, deviation = normalise_self.get_deviation(variable_name, spatial=True)

                    dataset[variable_name] = (dataset[variable_name] - average[variable_name]) / deviation[
                        variable_name
                    ]

                return dataset

        return DeviationNormaliser()

    @property
    def log(normalise_self):
        """
        Normalise with log
        """

        class LogNormaliser(Transform):
            """Normalise Dataset with log"""

            @property
            def _info_(self):
                return normalise_self._info_

            def apply(self, dataset: xr.Dataset):
                dataset = xr.Dataset(dataset)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for variable_name in dataset:
                        dataset[variable_name] = np.log(dataset[variable_name])
                    dataset = dataset.where(dataset.apply(np.isfinite)).fillna(0.0)
                return dataset

        return LogNormaliser()

    @property
    def function(self) -> FunctionTransform:
        """
        Get Transform from user provided function for functional normalisation

        Returns:
            FunctionTransform: Transform to apply Function
        """
        if self._function is None:
            raise ValueError("Cannot use function transform without a given function.\nTry giving `function` to init")
        return FunctionTransform(self._function.normalise)
