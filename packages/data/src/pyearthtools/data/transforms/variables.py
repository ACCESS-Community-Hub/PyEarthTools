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

import xarray as xr

from pyearthtools.data.transforms.transform import Transform


__all__ = ["Trim", "Drop"]


class Trim(Transform):
    """Trim dataset variables"""

    def __init__(self, variables: list[str] | str, *extra_variables):
        """
        Trim Dataset to given variables.

        If no variables would be left, apply no Transform

        Args:
            variables (list[str] | str):
                List of vars to trim to
        """
        super().__init__()
        self.record_initialisation()

        variables = variables if isinstance(variables, (list, tuple)) else [variables]
        self._variables = [*variables, *extra_variables]

    def apply(self, dataset: xr.Dataset) -> xr.Dataset:
        if self._variables is None:
            return dataset
        var_included = set(self._variables) & set(dataset.data_vars)
        if not var_included:
            return dataset
        return dataset[var_included]


class Drop(Transform):
    """Drop dataset variables"""

    def __init__(self, variables: list[str] | str, *extra_variables, **kwargs):
        """
        Drop variables from dataset

        Args:
            variables (list[str] | str):
                List of vars to drop
            kwargs (dict):
                kwargs to pass to drop
        """
        super().__init__()
        self.record_initialisation()

        variables = variables if isinstance(variables, (list, tuple)) else [variables]
        self._variables = [*variables, *extra_variables]
        self.kwargs = kwargs

    def apply(self, dataset: xr.Dataset) -> xr.Dataset:
        if self._variables is None:
            return dataset

        # 3/9/2025 - old logic was replaced with a simple drop of the variables
        # A new issue will be raised to review how coordinate protection should
        # work because people need a way to drop coords when needed.

        # Calculate the difference between the data variables on the dataset
        # and the variables requested for drop. This leaves coordinate variables
        # unaffected
        # var_included = set(dataset.data_vars).difference(set(self._variables))

        # if not var_included:
        #     return dataset
        # return dataset[var_included]
        return dataset.drop_vars(self._variables, **self.kwargs)


class Select(Transform):
    """Select specific dataset variables"""

    def __init__(self, variables: list[str] | str, *extra_variables):
        """
        Select variables from the dataset.

        Args:
            variables (list[str] | str):
                List of variables to select.
        """
        super().__init__()
        self.record_initialisation()

        # Ensure variables is always a list
        variables = variables if isinstance(variables, (list, tuple)) else [variables]
        self._variables = [*variables, *extra_variables]

    def apply(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Apply the transform to select specific variables.

        Args:
            dataset (xr.Dataset): The dataset to transform.

        Returns:
            xr.Dataset: A dataset containing only the selected variables.
        """
        if self._variables is None:
            return dataset

        # Select only the variables that exist in the dataset
        var_included = set(self._variables) & set(dataset.data_vars)

        if not var_included:
            # If no variables match, return the original dataset
            return dataset

        return dataset[list(var_included)]
