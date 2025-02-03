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

import pyearthtools.data.transforms.attributes as attr
from pyearthtools.data.transforms.transform import Transform

from pyearthtools.utils.decorators import BackwardsCompatibility

# Backwards compatability
rename_variables = attr.rename
replace_name_deviation = rename_variables


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


@BackwardsCompatibility(Trim)
def trim(*args) -> Transform: ...
@BackwardsCompatibility(Trim)
def variable_trim(*args) -> Transform: ...


class Drop(Transform):
    """Drop dataset variables"""

    def __init__(self, variables: list[str] | str, *extra_variables):
        """
        Drop variables from dataset

        Args:
            variables (list[str] | str):
                List of vars to drop
        """
        super().__init__()
        self.record_initialisation()

        variables = variables if isinstance(variables, (list, tuple)) else [variables]
        self._variables = [*variables, *extra_variables]

    def apply(self, dataset: xr.Dataset) -> xr.Dataset:
        if self._variables is None:
            return dataset

        var_included = set(dataset.data_vars).difference(set(self._variables))

        if not var_included:
            return dataset
        return dataset[var_included]


@BackwardsCompatibility(Drop)
def drop(*args) -> Transform: ...
