from typing import Union

import einops
import numpy as np

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
        value: int = 0,
        apply_iterator: bool = True,
        apply_get: bool = True,
    ) -> None:
        """
        Fill Nan's with Value

        Parameters
        ----------
        iterator
            Underlying Iterator
        value, optional
            Value to fill Nan with, by default 0
        apply_iterator, optional
            Whether to apply on __iter__, by default True
        apply_get, optional
            Whether to apply on __getitem__, by default True
        """
        super().__init__(iterator)
        self.value = value
        self.apply_iterator = apply_iterator
        self.apply_get = apply_get

        self.__doc__ = f"Fill nan's with {value}"
        if apply_iterator & apply_get:
            self.__doc__ += " on both iteration and get"
        elif apply_iterator ^ apply_get:
            self.__doc__ += f" on {'iteration' if apply_iterator else 'getitem'}"

    def _apply_fill(self, data):
        if isinstance(data, tuple):
            return tuple(map(lambda x: np.nan_to_num(x, self.value), data))
        return np.nan_to_num(data, self.value)

    def __iter__(self):
        for data in self.iterator:
            if self.apply_iterator:
                yield self._apply_fill(data)
            else:
                yield data

    def __getitem__(self, idx):
        if self.apply_get:
            return self._apply_fill(self.iterator[idx])
        return self.iterator[idx]


@SequentialIterator
class Rearrange(DataOperation):
    """
    Rearrange Data
    """

    def __init__(self, iterator: DataIterator, rearrange: str, *rearrange_args) -> None:
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
        *rearrange_args
            All to be passed to einops.rearrange

        """
        super().__init__(iterator)
        self.rearrange = rearrange
        self.rearrange_args = rearrange_args

        self.__doc__ = f"Rearrange Data according to {rearrange}"

    def __getitem__(self, idx):
        return self._apply_rearrange(self.iterator[idx], self.rearrange)

    def _apply_rearrange(
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
                raise excep
            rearrange = "->".join(["p " + side for side in rearrange.split("->")])
            return self._apply_rearrange(data, rearrange, catch=False)

    def __iter__(self):
        for data in self.iterator:
            yield self._apply_rearrange(data, self.rearrange)

    def undo(self, data, *args, **kwargs):
        reversed_rearrange = self.rearrange.split("->")
        reversed_rearrange.reverse()
        data = self._apply_rearrange(data, "->".join(reversed_rearrange))

        return super().undo(data, *args, **kwargs)
