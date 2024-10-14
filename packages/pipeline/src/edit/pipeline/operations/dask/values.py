# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

# type: ignore[reportPrivateImportUsage]

from typing import Literal, Union, Optional

import dask.array as da
import numpy as np

import edit.data

from edit.pipeline.operations.dask.dask import DaskOperation


class FillNan(DaskOperation):
    """
    Fill any Nan's with a value
    """

    _override_interface = ["Serial"]
    _numpy_counterpart = "values.FillNan"

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
        raise NotImplementedError("Not implemented")

        super().__init__(
            operation="apply",
            split_tuples=True,
            recursively_split_tuples=True,
            recognised_types=da.Array,
        )

        self.record_initialisation()

        self.nan = nan
        self.posinf = posinf
        self.neginf = neginf

    def apply_func(self, sample: da.Array):
        return da.nan_to_num(da.array(sample), self.nan, self.posinf, self.neginf)


class MaskValue(DaskOperation):
    """
    DataOperation to mask values with a given replacement


    """

    _override_interface = ["Serial"]
    _numpy_counterpart = "values.MaskValue"

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
            recognised_types=da.Array,
        )

        self.record_initialisation()

        if operation not in ["==", ">", "<", ">=", "<="]:
            raise KeyError(f"Invalid operation {operation!r}. Must be one of  ['==', '>', '<', '>=', '<=']")
        self.operation = operation
        self.value = value
        self.replacement_value = replacement_value

        self._mask_transform = edit.data.transforms.mask.replace_value(value, operation, replacement_value)

    def apply_func(self, sample: da.Array) -> da.Array:
        """
        Mask Data from initialised configuration

        Args:
            sample (da.Array):
                Data to apply mask to

        Returns:
            (da.Array):
                Masked Data
        """
        return self._mask_transform(sample)  # type: ignore


class ForceNormalised(DaskOperation):
    """
    Operation to force data within a certain range, by default 0 & 1
    """

    _override_interface = ["Serial"]
    _numpy_counterpart = "values.ForceNormalised"

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
            recognised_types=da.Array,
        )

        self.record_initialisation()

        self._force_min = MaskValue(min_value, "<", min_value) if min_value is not None else None
        self._force_max = MaskValue(max_value, ">", max_value) if max_value is not None else None

    def apply_func(self, sample):
        for func in (func for func in [self._force_min, self._force_max] if func is not None):
            sample = func.apply_func(sample)
        return sample
