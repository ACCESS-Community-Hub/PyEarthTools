from typing import Union

import numpy as np
import xarray as xr

from edit.training.data.templates import (
    DataIterator,
    DataStep,
    DataOperation,
)
from edit.training.data.sequential import Sequential, SequentialIterator


@SequentialIterator
class FillNa(DataOperation):
    """
    Fill nan's with value
    """

    def __init__(
        self,
        index: DataStep | DataIterator,
        nan: float = 0,
        posinf: float = None,
        neginf: float = None,
        **kwargs,
    ) -> None:
        """
        Fill Nan's with Value

        Parameters
        ----------
        index
            Underlying index
        nan, optional
            Value to fill Nan with, by default 0
            If no value is passed then NaN values will not be replaced
        posinf, optional
            Value to be used to fill positive infinity values, by default None
            If no value is passed then positive infinity values will be replaced with a very large number.
        neginf, optional
           Value to be used to fill negative infinity values, by default None
            If no value is passed then negative infinity values will be replaced with a very small (or negative) number.
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

    def _apply_fill(self, data):
        if isinstance(data, tuple):
            return tuple(map(self._apply_fill, data))
        if isinstance(data, (xr.Dataset, xr.DataArray)):
            data = data.fillna(self.nan)
        return np.nan_to_num(data, self.nan, posinf=self.posinf, neginf=self.neginf)

@SequentialIterator
class MaskValue(DataOperation):
    """
    Mask out Values given operation
    """
    def __init__(self, index: DataStep | DataIterator, value: int, operation: str, replacement_value: int = np.nan):
        """Mask out a value from Data

        Args:
            index (DataStep | DataIterator): 
                _description_
            value (int): 
                _description_
            operation (Literal["==", ">", "<", ">=", "<="], optional): 
                Criteria to search by. Defaults to "==".
            replacement_value (int, optional):
                Value to replace with. Defaults to np.nan
        """
        super().__init__(index, apply_func=self._mask, undo_func=None)
        if operation not in ["==", ">", "<", ">=", "<="]:
            raise KeyError(
                f"Invalid operation {operation!r}. Must be one of  ['==', '>', '<', '>=', '<=']"
            )
        self.operation = operation
        self.value = value
        self.replacement_value = replacement_value

        self.__doc__ = f"Given data {operation} {value} replace with {replacement_value}"
        if self.apply_iterator & self.apply_get:
            self.__doc__ += " on both iteration and get"
        elif self.apply_iterator ^ self.apply_get:
            self.__doc__ += f" on {'iteration' if self.apply_iterator else 'getitem'}"

    def _mask(self, data: xr.Dataset | np.ndarray | tuple):

        operator_package = np
        if isinstance(data, (xr.Dataset, xr.DataArray)):
            operator_package = xr

        if isinstance(data, tuple):
            return tuple(map(self._mask, data))

        if self.operation == "==":
            return operator_package.where(data == self.value, self.replacement_value, data)
        elif self.operation == ">":
            return operator_package.where(data > self.value, self.replacement_value, data)
        elif self.operation == ">=":
            return operator_package.where(data >= self.value, self.replacement_value, data)
        elif self.operation == "<":
            return operator_package.where(data < self.value, self.replacement_value, data)
        elif self.operation == "<=":
            return operator_package.where(data <= self.value, self.replacement_value, data)
        raise KeyError(f"Invalid operation {self.operation!r}")

@SequentialIterator
class ForceNormalised(DataOperation):
    """
    Force Data to be within 0 & 1
    """
    def __init__(self, index) -> None:
        super().__init__(index, apply_func=self._mask)

        self._force_min_0 = MaskValue(index, 0, '<', 0)
        self._force_max_1 = MaskValue(index, 1, '>', 1)

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
