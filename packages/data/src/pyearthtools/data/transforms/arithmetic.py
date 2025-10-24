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


"""
Basic arithmetic transformations
"""

import xarray as xr

from pyearthtools.data.transforms import Transform


class AddConstant(Transform):
    """Add a constant to a variable in dataset."""

    def __init__(
        self,
        **summands: dict[str, int | float],
    ):
        """
        Add a constant to a variable in dataset.

        Args:
            summands (dict):
                A dictionary of variable names and constant pairings
                (e.g., {'name':1}, will lead to a constant of 1 being added to a variable with 'name').
        """
        super().__init__()
        self.record_initialisation()

        self._summands = summands

    def apply(self, dataset: xr.Dataset) -> xr.Dataset:
        for key in self._summands.keys():
            try:
                dataset[key] = dataset[key] + self._summands[key]

            except KeyError as e:
                raise DataNotFoundError(f"Variable with name {key} not found in dataset!") from e
        return dataset


class SubtractConstant(Transform):
    """Subtract a constant from a variable in dataset."""

    def __init__(
        self,
        **subtrahends: dict[str, int | float],
    ):
        """
        Subtract a constant from a variable in dataset.

        Args:
            subtrahends (dict):
                A dictionary of variable names and constant pairings
                (e.g., {'name':1}, will lead to a constant of 1 being subtracted from a variable with 'name').
        """
        super().__init__()
        self.record_initialisation()

        self._subtrahends = subtrahends

    def apply(self, dataset: xr.Dataset) -> xr.Dataset:
        for key in self._subtrahends.keys():
            try:
                dataset[key] = dataset[key] - self._subtrahends[key]

            except KeyError as e:
                raise DataNotFoundError(f"Variable with name {key} not found in dataset!") from e
        return dataset


class MultiplyConstant(Transform):
    """Multiply variable in dataset with a constant."""

    def __init__(
        self,
        **factors: dict[str, int | float],
    ):
        """
        Multiply all values of a variable with a constant factor.

        Args:
            factors (dict):
                A dictionary of variable names and factor value pairings
                (e.g., {'name':2}, will lead to variable with 'name' to be doubled).
        """
        super().__init__()
        self.record_initialisation()

        self._factors = factors

    def apply(self, dataset: xr.Dataset) -> xr.Dataset:
        for key in self._factors.keys():
            try:
                dataset[key] = dataset[key] * self._factors[key]

            except KeyError as e:
                raise DataNotFoundError(f"Variable with name {key} not found in dataset!") from e
        return dataset


class DivideConstant(Transform):
    """Divide variable in dataset with a constant."""

    def __init__(
        self,
        **divisors: dict[str, int | float],
    ):
        """
        Divide all values of a variable with a constant factor.

        Args:
            divisors:
                A dictionary of variable names and divisor value pairings
                (e.g., {'name':2}, will lead to variable with 'name' to be halved).
        """
        super().__init__()
        self.record_initialisation()

        self._divisors = divisors

    def apply(self, dataset: xr.Dataset) -> xr.Dataset:
        for key in self._divisors.keys():
            try:
                dataset[key] = dataset[key] / self._divisors[key]

            except KeyError as e:
                raise DataNotFoundError(f"Variable with name {key} not found in dataset!") from e
        return dataset


class AddDataArray(Transform):
    """Add two variables from the same dataset."""

    def __init__(
        self,
        **summands: str,
    ):
        """
        Add two variables from the same dataset.

        Args:
            summands:
                A dictionary of summands. The key of the dict is the variable name of the
                first summand and result, while the value is the second summand.
                Follows: dataset[key] = dataset[key] + dataset[value].
        """
        super().__init__()
        self.record_initialisation()

        self._summands = summands

    def apply(self, dataset: xr.Dataset) -> xr.Dataset:
        for key in self._summands.keys():
            try:
                dataset[key] = dataset[key] + dataset[self._summands[key]]

            except KeyError as e:
                raise DataNotFoundError(
                    f"Variables with names {key} or {self._summands[key]} not found in dataset!"
                ) from e
        return dataset


class SubtractDataArray(Transform):
    """Subtract two variables from the same dataset."""

    def __init__(
        self,
        **subtrahends: str,
    ):
        """
        Subtract two variables from the same dataset.

        Args:
            subtrahends:
                A dictionary of subtrahends. The key of the dict is the variable name of the
                minuend and result, while the value is the subtrahend.
                Follows: dataset[key] = dataset[key] - dataset[value].
        """
        super().__init__()
        self.record_initialisation()

        self._subtrahends = subtrahends

    def apply(self, dataset: xr.Dataset) -> xr.Dataset:
        for key in self._subtrahends.keys():
            try:
                dataset[key] = dataset[key] - dataset[self._subtrahends[key]]

            except KeyError as e:
                raise DataNotFoundError(
                    f"Variables with names {key} or {self._subtrahends[key]} not found in dataset!"
                ) from e
        return dataset


class MultiplyDataArray(Transform):
    """Multiply two variables from the same dataset."""

    def __init__(
        self,
        **factors: str,
    ):
        """
        Multiply two variables from the same dataset.

        Args:
            factors:
                A dictionary of factors. The key of the dict is the variable name of the
                variable to be multipled and result, while the value is the factor.
                Follows: dataset[key] = dataset[key] * dataset[value].
        """
        super().__init__()
        self.record_initialisation()

        self._factors = factors

    def apply(self, dataset: xr.Dataset) -> xr.Dataset:
        for key in self._factors.keys():
            try:
                dataset[key] = dataset[key] * dataset[self._factors[key]]

            except KeyError as e:
                raise DataNotFoundError(
                    f"Variables with names {key} or {self._factors[key]} not found in dataset!"
                ) from e
        return dataset


class DivideDataArray(Transform):
    """Divide two variables from the same dataset."""

    def __init__(
        self,
        **divisors: str,
    ):
        """
        Divide two variables from the same dataset.

        Args:
            factors:
                A dictionary of divisors. The key of the dict is the variable name of the
                dividend and result, while the value is the divisor.
                Follows: dataset[key] = dataset[key] / dataset[value].
        """
        super().__init__()
        self.record_initialisation()

        self._divisors = divisors

    def apply(self, dataset: xr.Dataset) -> xr.Dataset:
        for key in self._divisors.keys():
            try:
                dataset[key] = dataset[key] / dataset[self._divisors[key]]

            except KeyError as e:
                raise DataNotFoundError(
                    f"Variables with names {key} or {self._divisors[key]} not found in dataset!"
                ) from e
        return dataset
