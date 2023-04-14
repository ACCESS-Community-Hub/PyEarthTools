import functools
from datetime import datetime, timedelta
from typing import Iterable, Union


import xarray as xr

import dset.data
from dset.data import FunctionTransform, Transform
from dset.data.default import DataIndex, OperatorIndex
from dset.data.time import DSETDatetime
from dset.data.transform import normalisation

from dset.training.data.templates import DataInterface, SequentialIterator, TrainingOperatorIndex



@SequentialIterator
class NormaliseInterface(DataInterface):
    """Normalisation DataInterface"""
    def __init__(
        self,
        index: Union[DataIndex, OperatorIndex, TrainingOperatorIndex],
        start: str,
        end: str,
        interval: Union[int, tuple[int, str]],
        method: Union[str, dict] = None,
        default: str = None,
        cache_dir: str = None,
        **kwargs,
    ) -> None:

        super().__init__(index, **kwargs)

        self.normalisation_params = dict(start = start, end = end, interval = interval, cache_dir = cache_dir)
        self.method = method
        self.default = default

    def get(self, querytime: Union[str, DSETDatetime]):
        return self._normalise(super().get(querytime))

    def normalise(
        self, dataset: Union[xr.Dataset, tuple[xr.Dataset]]
    ) -> Union[xr.Dataset, tuple[xr.Dataset]]:
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
        self, dataset: Union[xr.Dataset, tuple[xr.Dataset]]
    ) -> Union[xr.Dataset, tuple[xr.Dataset]]:
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

        return normalisation.unnormalise(self.index, **params)(self.method, self.default)

    @property
    def __doc__(self):
        return f"Normalising with method: {self.method} and default: {self.default}"


