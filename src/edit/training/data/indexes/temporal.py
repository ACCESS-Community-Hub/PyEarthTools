
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, Union

import xarray as xr

from edit.data import FunctionTransform, Transform, TransformCollection
from edit.data import DataIndex, OperatorIndex
from edit.data.time import EDITDatetime, TimeDelta

from edit.training.data.templates import DataIterator, DataStep, DataInterface, TrainingOperatorIndex, TrainingDataIndex
from edit.training.data.sequential import Sequential, SequentialIterator
from edit.training.data.utils import get_transforms

@SequentialIterator
class TemporalIndex(TrainingDataIndex):
    """
    TemporalIndex to provide capability to add a temporal dimension to loaded data.

    !!! Warning
        Must exist above a DataStep which still returns a [Dataset][xarray.Dataset]

    !!! Example
        ```python
        TemporalIndex(PipelineStep)

        ## As this is decorated with @SequentialIterator, it can be partially initialised

        partialTemporalIndex = TemporalIndex()
        partialTemporalIndex(PipelineStep)
        ```
    """    
    def __init__(
        self,
        index: DataStep,
        *,
        transforms: list[TransformCollection] | TransformCollection | str | dict = TransformCollection(),
        samples: tuple[int] | int = 1,
        sample_interval: int | tuple = [60, 'min'],
        use_safe_series: bool = True,
        **kwargs,
    ) -> None:
        """TemporalIndex to add a time dimension to the loaded data.
        
        
        Args:
            index (DataStep): 
                Underlying Pipeline step, must return a [Dataset][xarray.Dataset]
            transforms (list[TransformCollection] | TransformCollection | str | dict, optional): 
                Extra transforms to add to the retrieval of data. Defaults to None.
            samples (tuple[int] | int, optional): 
                Temporal Samples to retrieve, if tuple [prior,post], if int post. Defaults to [60, 'min'].
            sample_interval (int | tuple, optional): 
                Interval between samples, must be in form of [TimeDelta][edit.data.time.TimeDelta].
                If int, default to minutes unit. Defaults to 1.
            use_safe_series (bool, optional):

            **kwargs (Any, Optional):
                Extra kwargs to be passed to the retrieval calls.
        
        Raises:
            ValueError: 
                If `samples` and `sample_interval` are invalid
        """    

        super().__init__(index)

        self.retrieval_kwargs = kwargs
        self.transforms = get_transforms(transforms)
        self.use_safe_series = use_safe_series

        if isinstance(samples, list):
            samples = tuple(samples)

        self.samples = samples
        if isinstance(sample_interval, tuple) and isinstance(sample_interval[0], tuple):
            self.sample_interval = tuple(map(TimeDelta, sample_interval))
        else:
            self.sample_interval = TimeDelta(sample_interval)

        self._info_ = dict(samples = samples, sample_interval = sample_interval)

    def rebuild_time(
        self,
        dataset: tuple[xr.Dataset] | xr.Dataset,
        time_value: EDITDatetime | datetime | str,
        offset: int = 0,
        interval_index: int = -1
    ) -> tuple[xr.Dataset] | xr.Dataset:
        """Rebuild time dimension of given dataset, using known sample interval.

        Should be used after a destructive process in an iterator above, 
        that cannot restore the time dimension. 
        
        If 'time' dimension is not present, return dataset unchanged.
        
        
        Args:
            dataset (tuple[xr.Dataset] | xr.Dataset): 
                Dataset to rebuild
            time_value (EDITDatetime | datetime | str): 
                First timestep to use and thus iterate from
            offset (int, optional): 
                Offset to add to time in multiples of `sample_interval`. Defaults to 0.
            interval_index (int, optional):
                Sample Interval to use if `sample_interval` is a tuple. Defaults to -1.
        
        Returns:
            (tuple[xr.Dataset] | xr.Dataset): 
                Dataset/s with time dimensions rebuilt
        """        
        if isinstance(dataset, tuple):
            return tuple(map(self.rebuild_time, dataset))

        if "time" not in dataset:
            return dataset

        time_size = len(dataset["time"])
        time_value = EDITDatetime(time_value)

        interval = self.sample_interval
        if isinstance(self.sample_interval, tuple):
            interval = self.sample_interval[interval_index]

        new_time = [
            (time_value + interval * (i + offset)).datetime64()
            for i in range(time_size)
        ]
        return dataset.assign_coords(time=new_time)

    def _retrieve_from_index(
        self,
        timestep: str | datetime | EDITDatetime,
        index: DataIndex | OperatorIndex,
        transforms: TransformCollection,
    ) -> tuple:
        """
        At given index retrieve given time with samples
        """

        timestep = EDITDatetime(timestep)

        retrieval_func = index.safe_series if self.use_safe_series else index.series

        if self.samples == 1:
            data = index.single(
                timestep, transforms=transforms, **self.retrieval_kwargs
            )
            return (data,) if not isinstance(data, tuple) else data

        elif (
            isinstance(self.samples, int)
            and self.samples > 1
            and isinstance(index, OperatorIndex)
        ):
            data = retrieval_func(
                timestep,
                timestep + self.sample_interval * (self.samples),
                interval=self.sample_interval,
                transforms=transforms,
                chunks=self.retrieval_kwargs.pop("chunks", "auto"),
                verbose=self.retrieval_kwargs.pop("verbose", False),
                **self.retrieval_kwargs,
            )
            return (data,) if not isinstance(data, tuple) else data

        elif isinstance(self.samples, tuple):
            if isinstance(index, OperatorIndex) or True:
                interval = self.sample_interval
                if isinstance(self.sample_interval, tuple):
                    interval = self.sample_interval[0]

                data_prior = retrieval_func(
                    timestep - interval * self.samples[0],
                    timestep,
                    interval=interval,
                    transforms=transforms,
                    chunks=self.retrieval_kwargs.pop("chunks", "auto"),
                    verbose=self.retrieval_kwargs.pop("verbose", False),
                    **self.retrieval_kwargs,
                )
                if isinstance(self.sample_interval, tuple):
                    interval = self.sample_interval[1]

                data_next = retrieval_func(
                    timestep,  # + self.sample_interval,
                    timestep + interval * self.samples[1],
                    interval=interval,
                    transforms=transforms,
                    chunks=self.retrieval_kwargs.pop("chunks", "auto"),
                    verbose=self.retrieval_kwargs.pop("verbose", False),
                    **self.retrieval_kwargs,
                )
                return data_prior, data_next
            else:
                raise NotImplementedError("TODO: Samples for non OperatorIndex")

        else:
            raise NotImplementedError()

    # def _retrieve_at_time(self, timestep: str, datetime | EDITDatetime):
    #     """
    #     Retrieve Data from all DataIndexes at given time
    #     """

    #     return_list = []
    #     for i, index in enumerate(self.index):
    #         transforms = TransformCollection(self.transforms[i])

    #         for element in self._retrieve_from_index(
    #             timestep,
    #             index,
    #             transforms=transforms,
    #         ):
    #             return_list.append(element)

    #     if len(return_list) == 1:
    #         return return_list[0]
    #     else:
    #         return (*tuple(return_list),)


    def get(self, querytime: str | EDITDatetime):
        return self._retrieve_from_index(
                EDITDatetime(querytime), self.index, self.transforms
            )

    # def __getitem__(self, idx):
    #     if isinstance(idx, int):
    #         return self.index[idx]
    #     elif isinstance(idx, (str, EDITDatetime, datetime)):
    #         return self._retrieve_from_index(
    #             EDITDatetime(idx), self.index, self.transforms
    #         )
    #     elif isinstance(idx, tuple):
    #         next_idx = idx[1:]
    #         if len(next_idx) == 1:
    #             next_idx = next_idx[0]
    #         return self[idx[0]].__getitem__(next_idx)

    #     return self.index[idx]

    @property
    def __doc__(self):
        return f"Providing {self.samples!r} samples of {self.index.__class__.__name__!r}."

    def apply(self, data):
        if isinstance(data, tuple):
            return tuple(map(self.apply, data))

        if hasattr(self.index, "apply"):
            data = self.index.apply(data)
        return data

    def undo(self, data):
        if isinstance(data, tuple):
            return tuple(map(self.undo, data))

        if hasattr(self.index, "undo"):
            data = self.index.undo(data)
        return data

    def ignore_sanity(self):
        return False