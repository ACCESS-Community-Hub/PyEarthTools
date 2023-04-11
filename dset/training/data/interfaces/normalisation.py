import functools
from datetime import datetime, timedelta
from typing import Iterable, Union


import xarray as xr

import dset.data
from dset.data import FunctionTransform, Transform
from dset.data.default import DataIndex, OperatorIndex
from dset.data.time import dset_datetime
from dset.data.transform import normalisation

from dset.training.data.templates import DataInterface, SequentialIterator, TrainingOperatorIndex



@SequentialIterator
class NormaliseInterface(DataInterface):
    """Normalisation DataInterface"""
    def __init__(
        self,
        index: Union[DataIndex, OperatorIndex, TrainingOperatorIndex],
        normalisation_params: dict = {},
        **kwargs,
    ) -> None:

        super().__init__(index, **kwargs)

        self.normalisation_params = normalisation_params

    def get(self, querytime: Union[str, dset_datetime]):
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

        params = dict(self.normalisation_params)
        method = params.pop("method", 'None')
        default = params.pop("default", 'None')

        return normalisation.normalise(self.index, **params)(method, default)

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

        params = dict(self.normalisation_params)
        method = params.pop("method", 'None')
        default = params.pop("default", 'None')

        return normalisation.unnormalise(self.index, **params)(method, default)

    @property
    def __doc__(self):
        return f"Normalising with method: {self.normalisation_params.pop('method', 'None')}"


