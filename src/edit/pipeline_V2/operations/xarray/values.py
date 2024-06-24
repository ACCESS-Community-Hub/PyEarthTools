# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from typing import TypeVar, Union, Optional, Any, Literal

import xarray as xr
import numpy as np

import edit.data

from edit.pipeline_V2.operation import Operation

T = TypeVar("T", xr.Dataset, xr.DataArray)


class FillNan(Operation):
    """
    Fill any Nan's with a value
    """

    _override_interface = "Serial"

    def __init__(
        self,
        nan: float = 0,
        posinf: Optional[float] = None,
        neginf: Optional[float] = None,
    ) -> None:
        """
        DataOperation to fill Nan's

        Args:
            nan (float, optional):
                Value to fill nan's with.
                If no value is passed then NaN values will not be replaced. Defaults to 0.
            posinf (float, optional):
                Value to be used to fill positive infinity values,
                If no value is passed then positive infinity values will be replaced with a very large number. Defaults to None.
            neginf (float, optional):
                Value to be used to fill negative infinity values,
                If no value is passed then negative infinity values will be replaced with a very small (or negative) number. Defaults to None.
        """

        super().__init__(
            operation="apply",
            split_tuples=True,
            recursively_split_tuples=True,
            recognised_types=(xr.Dataset, xr.DataArray),
        )

        self.record_initialisation()

        self.nan = nan
        self.posinf = posinf
        self.neginf = neginf

    def apply_func(self, sample: T) -> T:
        return sample.fillna(self.nan)


class MaskValue(Operation):
    """
    DataOperation to mask values with a given replacement
    """

    _override_interface = "Serial"

    def __init__(
        self,
        value: Union[float, dict[str, float]],
        operation: Literal["==", ">", "<", ">=", "<="] = "==",
        replacement_value: Union[float, dict[str, float]] = np.nan,
    ):
        """
        Operation to Mask Values

        Args:
            value (Union[float, dict[str, float]]):
                Value to search for.
            operation (Literal['==', '>', '<', '>=','<='], optional):
                Operation to search with. Defaults to '=='.
            replacement_value (Union[float, dict[str, float]], optional):
                Replacement value. Defaults to np.nan.

        Raises:
            KeyError:
                If invalid `operation` passed.
        """
        super().__init__(
            operation="apply",
            split_tuples=True,
            recursively_split_tuples=True,
            recognised_types=(xr.Dataset, xr.DataArray),
        )

        self.record_initialisation()

        if operation not in ["==", ">", "<", ">=", "<="]:
            raise KeyError(f"Invalid operation {operation!r}. Must be one of  ['==', '>', '<', '>=', '<=']")

        self.operation = operation
        self.value = value
        self.replacement_value = replacement_value

        self._mask_transform = edit.data.transforms.mask.replace_value(value, operation, replacement_value)

    def apply_func(self, sample: T) -> T:
        """
        Mask Data from initialised configuration

        Args:
            sample (T):
                Data to apply mask to

        Returns:
            (T):
                Masked Data
        """

        return self._mask_transform(sample)


class ForceNormalised(Operation):
    """
    Operation to force data within a certain range, by default 0 & 1
    """

    _override_interface = "Serial"

    def __init__(
        self,
        min_value: Optional[Union[float, dict[str, float]]] = 0,
        max_value: Optional[Union[float, dict[str, float]]] = 1,
    ):
        """
        Force data into a specified range

        Args:
            min_value (Optional[Union[float, dict[str, float]]], optional):
                Minimum Value. If using a dict, acts per variable given.
                If None, this won't apply a min masking
                Defaults to 0.
            max_value (Optional[Union[float, dict[str, float]]], optional):
                Maximum Value. If using a dict, acts per variable given.
                If None, this won't apply a max masking
                Defaults to 1.
        """
        super().__init__(
            operation="apply",
            split_tuples=True,
            recursively_split_tuples=True,
            recognised_types=(xr.Dataset, xr.DataArray),
        )

        self.record_initialisation()

        self._force_min = MaskValue(min_value, "<", min_value) if min_value is not None else None
        self._force_max = MaskValue(max_value, ">", max_value) if max_value is not None else None

    def apply_func(self, sample):
        for func in (func for func in [self._force_min, self._force_max] if func is not None):
            sample = func.apply_func(sample)
        return sample


class Derive(Operation):
    """
    Derive variables within the dataset

    Uses `edit.data.transforms.derive`.
    """

    _override_interface = "Serial"

    def __init__(
        self,
        derivation: Optional[dict[str, Union[str, tuple[str, dict[str, Any]]]]] = None,
        *,
        drop: bool = True,
        **derivations: Union[str, tuple[str, dict[str, Any]]],
    ):
        """
        Derivation step

        Args:
            derivation (Optional[dict[str, Union[str, tuple[str, dict[str, Any]]]]], optional):
                Equation configuration. If str, equation is evaluated.
                If tuple, first element is assumed to be equation, and the second a
                dictionary to update the new vars attributes with. Defaults to None.
            drop (bool, optional):
                Drop derived variables on `undo`. Defaults to True.
            **derivations (Union[str, tuple[str, dict[str, Any]]]):
                Kwarg form of `derivation`.
        """
        super().__init__(split_tuples=True, recursively_split_tuples=True, recognised_types=(xr.DataArray, xr.Dataset))
        self.record_initialisation()

        derivation = derivation or {}
        derivation.update(derivations)

        self._derive = edit.data.transforms.derive(derivation)
        self._drop = edit.data.transform.variables.drop(list(derivation.keys())) if drop else None

    def apply_func(self, sample):
        return self._derive(sample)

    def undo_func(self, sample):
        return self._drop(sample) if self._drop else sample
