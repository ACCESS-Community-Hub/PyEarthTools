import functools
from datetime import datetime, timedelta
from typing import Iterable, Union


import xarray as xr

import edit.data
from edit.data import FunctionTransform, Transform
from edit.data import DataIndex, OperatorIndex
from edit.data.time import EDITDatetime
from edit.data.transform import normalisation

from edit.training.data.templates import DataInterface, TrainingOperatorIndex
from edit.training.data.sequential import Sequential, SequentialIterator


@SequentialIterator
class NormaliseInterface(DataInterface):
    """Normalisation DataInterface"""

    def __init__(
        self,
        index: DataIndex | OperatorIndex | TrainingOperatorIndex,
        start: str,
        end: str,
        interval: int | tuple[int, str],
        method: str | dict = None,
        default: str = None,
        cache_dir: str = None,
        **kwargs,
    ) -> None:
        super().__init__(index, **kwargs)

        self.normalisation_params = dict(
            start=start, end=end, interval=interval, cache_dir=cache_dir
        )
        self.method = method
        self.default = default

    def get(self, querytime: str | EDITDatetime):
        return self._normalise(super().get(querytime))

    def normalise(
        self, dataset: xr.Dataset | tuple[xr.Dataset]
    ) -> xr.Dataset | tuple[xr.Dataset]:
        """
        Apply normalisation like done on retrieval but to given data

        Parameters
        ----------
        dataset
            Dataset/s to apply normalisation to

        Returns
        -------
            Normalised Dataset
        """
        return self._normalise(dataset)

    @property
    @functools.lru_cache(None)
    def _normalise(self) -> tuple[Transform]:
        """
        Get normalisation transforms
        """

        params = dict(**self.normalisation_params)

        return normalisation.normalise(self.index, **params)(self.method, self.default)

    def unnormalise(
        self, dataset: xr.Dataset | tuple[xr.Dataset]
    ) -> xr.Dataset | tuple[xr.Dataset]:
        """
        Apply Unnormalisation to given data.
        Can be used on direct output of self[]

        Parameters
        ----------
        dataset
            Dataset/s to apply unnormalisation to

        Returns
        -------
            Unnormalised Dataset

        Examples
        --------
            >>> MLIndex.unnormalise(MLIndex[date])
        """
        return self._unnormalise(dataset)

    def undo(self, *args, **kwargs):
        """
        Undo changes done to data
        """
        return super().undo(self.unnormalise(*args, **kwargs))

    @property
    def _unnormalise(self) -> tuple[Transform]:
        """
        Get unnormalisation transforms.
        """

        params = dict(**self.normalisation_params)

        return normalisation.unnormalise(self.index, **params)(
            self.method, self.default
        )

    @property
    def __doc__(self):
        return f"Normalising with method: {self.method} and default: {self.default}"
