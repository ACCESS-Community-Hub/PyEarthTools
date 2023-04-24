from typing import Union

import einops
import numpy as np
from scipy import interpolate

from dset.training.data.templates import (
    DataIterationOperator,
    DataIterator,
    DataOperation,
)
from dset.training.data.sequential import Sequential, SequentialIterator


@SequentialIterator
class Rearrange(DataOperation):
    """
    Rearrange Data
    """

    def __init__(
        self,
        index: DataIterator,
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
        index
            Iterator to use
        rearrange
            String entry to einops.rearrange
        skip
            Whether to skip data that cannot be rearranged
        *rearrange_args
            All to be passed to einops.rearrange

        """
        super().__init__(index, self._apply_rearrange, self._undo_rearrange, **kwargs)
        self.rearrange = rearrange
        self.rearrange_args = rearrange_args

        self.skip = skip
        self.__doc__ = f"Rearrange Data according to {rearrange}"

    def __rearrange(
        self, data: tuple[np.ndarray] | np.ndarray, rearrange: str, catch=True
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

    def __init__(self, index: DataIterator, axis: int, **kwargs) -> None:
        super().__init__(index, self._apply_squish, self._apply_expand, **kwargs)
        self.axis = axis

    def _apply_squish(self, data):
        if isinstance(data, tuple):
            return tuple(map(self._apply_squish, data))
        return np.squeeze(data, self.axis)

    def _apply_expand(self, data):
        if isinstance(data, tuple):
            return tuple(map(self._apply_squish, data))
        return np.expand_dims(data, self.axis)

@SequentialIterator
class Expand(DataOperation):
    """
    Expand One Dimensional axis at 'axis' location
    """

    def __init__(self, index: DataIterator, axis: int, **kwargs) -> None:
        super().__init__(index, self._apply_expand, self._apply_squish, **kwargs)
        self.axis = axis

    def _apply_squish(self, data):
        if isinstance(data, tuple):
            return tuple(map(self._apply_squish, data))
        return np.squeeze(data, self.axis)

    def _apply_expand(self, data):
        if isinstance(data, tuple):
            return tuple(map(self._apply_squish, data))
        return np.expand_dims(data, self.axis)