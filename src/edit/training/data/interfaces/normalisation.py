from __future__ import annotations

import functools


import xarray as xr

from edit.data import Transform
from edit.data import DataIndex, OperatorIndex
from edit.data.time import EDITDatetime
from edit.data.transform import normalisation

from edit.training.data.templates import DataInterface, TrainingOperatorIndex
from edit.training.data.sequential import SequentialIterator


@SequentialIterator
class NormaliseInterface(DataInterface):
    """
    Normalisation DataInterface which will automatically calculate and normalise data generated from the underlying [DataIndex][edit.data.DataIndex]

    Uses [edit.data.transform.normalisation][edit.data.transform.normalisation] to calculate the normalisation

    !!! Example
        ```python
        NormaliseInterface(PipelineStep, start = '2010', end = '2020', interval = [1,'day'], method = 'range')

        ## As this is decorated with @SequentialIterator, it can be partially initialised

        partialNormalise = NormaliseInterface(start = '2010', end = '2020', interval = [1,'day'], method = 'range')
        partialNormalise(PipelineStep)
        ```
    """

    def __init__(
        self,
        index: DataIndex | OperatorIndex | TrainingOperatorIndex,
        start: str | EDITDatetime,
        end: str | EDITDatetime,
        interval: int | tuple[int, str],
        method: str | dict = None,
        default: str = None,
        cache_dir: str = None,
        **kwargs,
    ) -> None:
        """A DataInterface to Normalise a given DataIndex as data is retrieved


        Args:
            index (DataIndex | OperatorIndex | TrainingOperatorIndex):
                Underlying DataIndex to retrieve Data from
            start (str | EDITDatetime):
                Date to start the normalisation search from
            end (str | EDITDatetime):
                Date to end the normalisation search from
            interval (int | tuple[int, str]):
                Interval at which to retrieve data at for the normalisation.
            method (str | dict, optional):
                Method to normalise with, can be dict assigning variables to methods. Defaults to None.
            default (str, optional):
                Default method if variable not found in dict method. Defaults to None.
            cache_dir (str, optional):
                Caching location for the normalisation values. Defaults to None.
        """
        super().__init__(
            index, apply_func=self.normalise, undo_func=self.unnormalise, **kwargs
        )

        self.normalisation_params = dict(
            start=start, end=end, interval=interval, cache_dir=cache_dir
        )
        self.method = method
        self.default = default

    def get(self, querytime: str | EDITDatetime) -> xr.Dataset:
        """Get normalise data at a given time

        Args:
            querytime (str | EDITDatetime):
                Time to retrieve data at

        Returns:
            (xr.Dataset):
                Normalised Data
        """
        return self._normalise(super().get(querytime))

    def normalise(self, dataset: xr.Dataset) -> xr.Dataset | tuple[xr.Dataset]:
        """Apply normalisation like done on retrieval but to given data

        Args:
            dataset (xr.Dataset):
                Dataset/s to apply normalisation to

        Returns:
            (xr.Dataset | tuple[xr.Dataset]):
                Normalised Dataset
        """
        return self._normalise(dataset.copy())

    @property
    @functools.lru_cache(None)
    def _normalise(self) -> tuple[Transform]:
        """
        Get normalisation transforms
        """

        params = dict(**self.normalisation_params)

        return normalisation.normalise(self.index, **params)(self.method, self.default)

    def unnormalise(self, dataset: xr.Dataset) -> xr.Dataset:
        """Apply Unnormalisation to given data

        Can be used on direct output of self[]

        Args:
            dataset (xr.Dataset):
                Dataset/s to apply unnormalisation to

        Returns:
            (xr.Dataset | tuple[xr.Dataset]):
                UnNormalised Dataset
        """
        return self._unnormalise(dataset.copy())

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
