# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations

import xarray as xr

import edit.data.transforms.attributes as attr
from edit.data.transforms.transform import Transform

from edit.utils.decorators import BackwardsCompatibility

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
