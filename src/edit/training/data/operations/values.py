
from __future__ import annotations
from typing import Literal

import numpy as np
import xarray as xr

from edit.training.data.templates import (
    DataIterator,
    DataStep,
    DataOperation,
)
from edit.training.data.sequential import Sequential, SequentialIterator


@SequentialIterator
class FillNan(DataOperation):
    """
    DataOperation to Fill any Nan's with a value

    !!! Example
        ```python
        FillNan(PipelineStep, nan = 0)

        ## As this is decorated with @SequentialIterator, it can be partially initialised

        partialFillNan = FillNan(nan = 0)
        partialFillNan(PipelineStep)
        ```
    """

    def __init__(
        self,
        index: DataStep,
        nan: float = 0,
        posinf: float = None,
        neginf: float = None,
        **kwargs,
    ) -> None:
        """
        DataOperation to fill Nan's

        Args:
            index (DataStep): 
                Underlying DataStep to retrieve Data from.
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

        self.nan = nan
        self.posinf = posinf
        self.neginf = neginf
        super().__init__(index, self._apply_fill, None, **kwargs)

        self.__doc__ = f"Fill nan's with {nan}"
        if self.apply_iterator & self.apply_get:
            self.__doc__ += " on both iteration and get"
        elif self.apply_iterator ^ self.apply_get:
            self.__doc__ += f" on {'iteration' if self.apply_iterator else 'getitem'}"
        
        self._info_ = dict(nan = nan, posinf = posinf, neginf = neginf)

    def _apply_fill(self, data):
        if isinstance(data, tuple):
            return tuple(map(self._apply_fill, data))
        if isinstance(data, (xr.Dataset, xr.DataArray)):
            data = data.fillna(self.nan)
        return np.nan_to_num(data, self.nan, posinf=self.posinf, neginf=self.neginf)

FillNa = FillNan

@SequentialIterator
class MaskValue(DataOperation):
    """
    DataOperation to mask values with a given replacement

    !!! Example
        ```python
        MaskValue(PipelineStep, value = 0, operation = '<', replacement_value = 0)

        ## As this is decorated with @SequentialIterator, it can be partially initialised

        partialMaskValue = MaskValue(value = 0, operation = '<', replacement_value = 0)
        partialMaskValue(PipelineStep)
        ```
    """

    def __init__(
        self,
        index: DataStep,
        value: int,
        operation: "Literal['==', '>', '<', '>=','<=']" = '==',
        replacement_value: int = np.nan,
    ):
        """
        DataOperation to Mask Values

        Args:
            index (DataStep): 
                Underlying DataStep to retrieve Data from.
            value (int): 
                Value to search for
            operation (Literal['==', '>', '<', '>=','<='], optional): 
                Operation to search with. Defaults to '=='.
            replacement_value (int, optional): 
                Replacement value. Defaults to np.nan.

        Raises:
            KeyError: 
                If invalid `operation` passed.
        """        
        super().__init__(index, apply_func=self._mask, undo_func=None)
        
        if operation not in ["==", ">", "<", ">=", "<="]:
            raise KeyError(
                f"Invalid operation {operation!r}. Must be one of  ['==', '>', '<', '>=', '<=']"
            )
        self.operation = operation
        self.value = value
        self.replacement_value = replacement_value

        self.__doc__ = (
            f"Given data {operation} {value} replace with {replacement_value}"
        )
        self._info_ = dict(value = value, operation = operation, replacement_value = replacement_value)

        if self.apply_iterator & self.apply_get:
            self.__doc__ += " on both iteration and get"
        elif self.apply_iterator ^ self.apply_get:
            self.__doc__ += f" on {'iteration' if self.apply_iterator else 'getitem'}"

    def _mask(self, data: xr.Dataset | np.ndarray | tuple) -> xr.Dataset | np.ndarray | tuple:
        """
        Mask Data from initialised configuration

        Args:
            data (xr.Dataset | np.ndarray | tuple): 
                Data to apply mask to

        Returns:
            (xr.Dataset | np.ndarray | tuple): 
                Masked Data
        """        
        operator_package = np
        if isinstance(data, (xr.Dataset, xr.DataArray)):
            operator_package = xr

        if isinstance(data, tuple):
            return tuple(map(self._mask, data))

        if self.operation == "==":
            return operator_package.where(
                data == self.value, self.replacement_value, data
            )
        elif self.operation == ">":
            return operator_package.where(
                data > self.value, self.replacement_value, data
            )
        elif self.operation == ">=":
            return operator_package.where(
                data >= self.value, self.replacement_value, data
            )
        elif self.operation == "<":
            return operator_package.where(
                data < self.value, self.replacement_value, data
            )
        elif self.operation == "<=":
            return operator_package.where(
                data <= self.value, self.replacement_value, data
            )
        raise KeyError(f"Invalid operation {self.operation!r}")


@SequentialIterator
class ForceNormalised(DataOperation):
    """
    DataOperation to force data within a certain range, by default 0 & 1

    !!! Example
        ```python
        ForceNormalised(PipelineStep)

        ## As this is decorated with @SequentialIterator, it can be partially initialised

        partialForceNormalised = ForceNormalised()
        partialForceNormalised(PipelineStep)
        ```
    """

    def __init__(self, index: DataStep, min_value: float = 0, max_value: float = 1) -> None:
        """
        DataOperation to force data into a range

        Args:
            index (DataStep): 
                Underlying DataStep to retrieve Data from.
            min_value (float, optional): 
                Minimum Value. Defaults to 0.
            max_value (float, optional): 
                Maximum Value. Defaults to 1.
        """        
        super().__init__(index, apply_func=self._mask, undo_func=None)

        self._force_min_0 = MaskValue(index, min_value, "<", min_value)
        self._force_max_1 = MaskValue(index, max_value, ">", max_value)

        self.__doc__ = f"Force Data between {min_value} and {max_value}"
        self._info_ = dict(min_value = min_value, max_value = max_value)

    def _mask(self, data):
        if isinstance(data, tuple):
            return tuple(map(self._mask, data))

        data = self._force_min_0._mask(data)
        data = self._force_max_1._mask(data)
        return data


# @SequentialIterator
# class InterpNan(DataOperation):
#     def __init__(
#         self,
#         iterator: DataIterator,
#         method: str = "linear",
#         **kwargs,
#     ) -> None:
#         """
#         Interpolate nan's in data using scipy.interpolate

#         Parameters
#         ----------
#         data
#             Data containing nan's to interpolate
#         method, optional
#             Method of interpolation to use, by default 'linear'

#         Returns
#         -------
#             Interpolated Data
#         """
#         self.method = method
#         super().__init__(iterator, self._apply_interpolation, None, **kwargs)

#     def _apply_interpolation(self, data):
#         if isinstance(data, tuple):
#             return tuple(map(self._apply_interpolation, data))

#         if len(data.shape) > 2:
#             indi_data = []
#             for channel in data:
#                 indi_data.append(self._apply_interpolation(channel))
#             return np.array(indi_data)

#         size = data.shape[-2:]
#         grid_x, grid_y = np.mgrid[
#             0 : size[0] : complex(size[0]), 0 : size[1] : complex(size[1])
#         ]

#         isnan = np.isnan(data)

#         points = np.array(np.where(~isnan)).T
#         points_to_interp = (grid_x, grid_y)
#         values = data[~isnan]

#         if points.size == 0:
#             return data

#         data = interpolate.griddata(
#             points, values, points_to_interp, method=self.method
#         )
#         return data
