import copy
import functools
from datetime import datetime, timedelta
import importlib
from typing import Iterable, Union

import builtins

import xarray as xr

import dset.data
from dset.data import FunctionTransform, Transform, TransformCollection
from dset.data.default import DataIndex, OperatorIndex
from dset.data.time import dset_datetime, time_delta

from dset.training.data.templates import DataIterator, SequentialIterator


@SequentialIterator
class TemporalInterface(DataIterator):
    """Base Temporal DataInterface"""
    def __init__(
        self,
        index: Union[list[DataIndex], DataIndex],
        transforms: Union[list[TransformCollection], TransformCollection] = None,
        samples: Union[tuple[int], int] = 1,
        sample_interval: Union[int, tuple] = 0,
        catch: Union[tuple[Exception], Exception] = None,
        **kwargs,
    ) -> None:
        """
        An extension of DataIndexes designed for ML Training,
        Using the provided index/s allows iteration between bound
        automatically applying a normalisation and other transforms.

        Also allows for multiple samples to be returned.

        Parameters
        ----------
        index
            DataIndex/s to use to retrieve data
        normalisation_params, optional
            Parameters for transform.normalise, as well as which method.
                If not given, no normalisation, by default None
            Params:
                start - start date for search
                end - end date for search
                interval - interval between searches
                cache_dir - Where to save Data used
                function - Function to use with functional normalisation

                method - which method to use, or dict assigning variable names to a method
                default - default method if above not found

        transforms, optional
            Other transforms to apply to data, can be list of transforms correspondant to data_indexes,
            by default None
        samples, optional
            Number of data samples to retrieve.
            If tuple[int,int], retrieve samples[0] before including querytime and samples[1] after,
            by default 1
        sample_interval, optional
            Interval between samples, by default 0
        **kwargs, optional
            All passed to data retrieval functions

        Raises
        ------
        ValueError
            If Transforms is list for each index but not the right size
        ValueError
            If samples > 1 and sample_interval == 0
        """

        # if not isinstance(index, (list, tuple)):
        #     index = [index]
        super().__init__(index, catch)

        self.retrieval_kwargs = kwargs
        self.transforms = transforms

        # if (
        #     isinstance(transforms, list)
        #     and len(transforms) > 0
        #     and isinstance(transforms, Iterable)
        # ):
        #     if not len(transforms) == len(index):
        #         raise ValueError(
        #             f"If transforms is list of seperate TransformCollections, must be same size as data_index. {len(index)} != {len(transforms)}"
        #         )
        #     self.transforms = transforms
        # else:
        #     self.transforms = [TransformCollection(transforms)] * len(index)

        if isinstance(samples, int) and samples > 1 and sample_interval == 0:
            raise ValueError(f"If 'samples' > 1, 'sample_interval' cannot be 0")
        if isinstance(samples, list):
            samples = tuple(samples)

        self.samples = samples
        self.sample_interval = time_delta(sample_interval)



    def rebuild_time(
        self,
        dataset: Union[tuple[xr.Dataset], xr.Dataset],
        time_value: Union[dset_datetime, datetime],
    ):
        """
        Rebuild time dimension of given dataset, using known sample interval.

        Should be used after a destructive process in an iterator above, 
        that cannot restore the time dimension. 
        
        If 'time' dimension is not present, return dataset unchanged.

        Parameters
        ----------
        dataset
            Dataset to rebuild
        time_value
            First timestep to use and thus iterate from

        Returns
        -------
            Dataset with time dimension rebuilt
        """ ""
        if isinstance(dataset, tuple):
            return tuple(map(self.rebuild_time, dataset))

        if not "time" in dataset:
            return dataset

        time_size = len(dataset["time"])
        time_value = dset_datetime(time_value)

        new_time = [
            (time_value + self.sample_interval * i).datetime64()
            for i in range(time_size)
        ]
        return dataset.assign_coords(time=new_time)

    def _retrieve_from_index(
        self,
        timestep: Union[str, datetime, dset_datetime],
        index: Union[DataIndex, OperatorIndex],
        transforms: TransformCollection,
    ) -> tuple:
        """
        At given index retrieve given time with samples
        """

        timestep = dset_datetime(timestep)

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
            data = index.safe_series(
                timestep,
                timestep + self.sample_interval * (self.samples - 1),
                interval = self.sample_interval,
                transforms=transforms,
                chunks=self.retrieval_kwargs.pop("chunks", "auto"),
                verbose=self.retrieval_kwargs.pop("verbose", False),
                **self.retrieval_kwargs,
            )
            return (data,) if not isinstance(data, tuple) else data

        elif isinstance(self.samples, tuple):
            if isinstance(index, OperatorIndex) or True:
                data_prior = index.safe_series(
                    timestep - self.sample_interval * max(0,(self.samples[0] - 1)),
                    timestep,
                    interval = self.sample_interval,
                    transforms=transforms,
                    chunks=self.retrieval_kwargs.pop("chunks", "auto"),
                    verbose=self.retrieval_kwargs.pop("verbose", False),
                    **self.retrieval_kwargs,
                )
                data_next = index.safe_series(
                    timestep + self.sample_interval,
                    timestep + self.sample_interval * self.samples[1],
                    interval = self.sample_interval,
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

    # def _retrieve_at_time(self, timestep: Union[str, datetime, dset_datetime]):
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

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.index[idx]
        elif isinstance(idx, (str, dset_datetime, datetime)):
            return self._retrieve_from_index(dset_datetime(idx), self.index, self.transforms)
        elif isinstance(idx, tuple):
            next_idx = idx[1:]
            if len(next_idx) == 1:
                next_idx = next_idx[0]
            return self[idx[0]].__getitem__(next_idx)
        raise ValueError



    def _formatted_name(self):
        desc = f"Data Interface for {self.index.__class__.__name__!r}. Providing {self.samples!r} samples"
        return super()._formatted_name(desc)


