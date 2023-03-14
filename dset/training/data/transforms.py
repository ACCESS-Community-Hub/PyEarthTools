from typing import Union

import einops
import numpy as np
from scipy import interpolate

from dset.training.data.templates import (
    DataIterationOperator,
    DataIterator,
    DataOperation,
    SequentialIterator,
)


@SequentialIterator
class FillNa(DataOperation):
    """
    Fill nan's with value
    """

    def __init__(
        self,
        iterator: DataIterator,
        nan: float = np.nan,
        posinf: float = None,
        neginf: float = None,
        **kwargs,
    ) -> None:
        """
        Fill Nan's with Value

        Parameters
        ----------
        iterator
            Underlying Iterator
        nan, optional
            Value to fill Nan with, by default NaN
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
        super().__init__(iterator, self._apply_fill, None, **kwargs)

        self.__doc__ = f"Fill nan's with {nan}"
        if self.apply_iterator & self.apply_get:
            self.__doc__ += " on both iteration and get"
        elif self.apply_iterator ^ self.apply_get:
            self.__doc__ += f" on {'iteration' if self.apply_iterator else 'getitem'}"

    def _apply_fill(self, data):
        if isinstance(data, tuple):
            return tuple(map(self._apply_fill, data))
        return np.nan_to_num(data, self.nan, posinf=self.posinf, neginf=self.neginf)

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

@SequentialIterator
class Rearrange(DataOperation):
    """
    Rearrange Data
    """

    def __init__(
        self,
        iterator: DataIterator,
        rearrange: str,
        skip: bool = False,
        *rearrange_args,
        **kwargs,
    ) -> None:
        """
        Using Einops rearrange, rearrange data.

        NOTE: This will occur on each iteration, and on __getitem__,
            so it is best to leave patches out if using PatchingDataIndex.

        Parameters
        ----------
        iterator
            Iterator to use
        rearrange
            String entry to einops.rearrange
        skip
            Whether to skip data that cannot be rearranged
        *rearrange_args
            All to be passed to einops.rearrange

        """
        super().__init__(iterator, self._apply_rearrange, self._undo_rearrange, **kwargs)
        self.rearrange = rearrange
        self.rearrange_args = rearrange_args

        self.skip = skip
        self.__doc__ = f"Rearrange Data according to {rearrange}"

    def __rearrange(
        self, data: Union[tuple[np.array], np.array], rearrange: str, catch=True
    ):
        """
        Apply einops.rearrange on data.

        If this fails, attempt to add 'p' to either side.
        """
        try:
            if isinstance(data, tuple):
                return tuple(
                    map(
                        lambda x: einops.rearrange(x, rearrange, *self.rearrange_args),
                        data,
                    )
                )
            return einops.rearrange(data, self.rearrange)

        except einops.EinopsError as excep:
            if not catch:
                if self.skip:
                    return data
                raise excep
            rearrange = "->".join(["p " + side for side in rearrange.split("->")])
            return self.__rearrange(data, rearrange, catch=False)

    def _apply_rearrange(self, data):
        return self.__rearrange(data, self.rearrange)

    def _undo_rearrange(self, data):
        reversed_rearrange = self.rearrange.split("->")
        reversed_rearrange.reverse()
        return self.__rearrange(data, "->".join(reversed_rearrange))

@SequentialIterator
class Squish(DataOperation):
    """
    Squish One Dimensional axis at 'axis' location
    """

    def __init__(self, iterator: DataIterator, axis: int, **kwargs) -> None:
        super().__init__(iterator, self._apply_squish, self._apply_expand, **kwargs)
        self.axis = axis

    def _apply_squish(self, data):
        if isinstance(data, tuple):
            return tuple(map(self._apply_squish, data))
        return np.squeeze(data, self.axis)

    def _apply_expand(self, data):
        if isinstance(data, tuple):
            return tuple(map(self._apply_squish, data))
        return np.expand_dims(data, self.axis)