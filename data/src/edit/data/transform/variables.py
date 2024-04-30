# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

from __future__ import annotations

import xarray as xr

import edit.data.transform.attributes as attr
from edit.data.transform.transform import Transform

# Backwards compatability
rename_variables = attr.rename
replace_name_deviation = rename_variables


def variable_trim(variables: list[str] | str, *extra_variables) -> Transform:
    """
    Trim Dataset to given variables.

    If no variables would be left, apply no Transform

    Args:
        variables (list[str] | str):
            List of vars to trim to

    Returns:
        Transform: Transform to trim dataset
    """
    variables = variables if isinstance(variables, (list, tuple)) else [variables]
    variables = [*variables, *extra_variables]

    class VariableTrim(Transform):
        """Trim Dataset to given variables"""

        @property
        def _info_(self):
            return dict(variables=variables)

        def apply(self, dataset: xr.Dataset) -> xr.Dataset:
            if variables is None:
                return dataset
            var_included = set(variables) & set(dataset.data_vars)
            if not var_included:
                return dataset
            return dataset[var_included]

    return VariableTrim()


def drop(variables: list[str] | str, *extra_variables) -> Transform:
    """Drop variables from dataset

    Args:
        variables (list[str] | str):
            Variables to drop

    Returns:
        (Transform):
            Transform to drop variables.
    """
    variables = variables if isinstance(variables, (list, tuple)) else [variables]
    variables = [*variables, *extra_variables]

    class DropVariables(Transform):
        """Drop variables"""

        @property
        def _info_(self):
            return dict(variables=variables)

        def apply(self, dataset: xr.Dataset) -> xr.Dataset:
            if variables is None:
                return dataset
            var_included = set(dataset.data_vars).difference(set(variables))
            if not var_included:
                return dataset
            return dataset[var_included]

    return DropVariables()
