import copy
import functools
from datetime import datetime, timedelta
from typing import Iterable, Union

import xarray as xr
from dset.data import FunctionTransform, Transform, TransformCollection
from dset.data.default import DataIndex, OperatorIndex
from dset.data.time import dset_datetime
from dset.data.transform import default, normalisation

from dset.training.data.templates import DataInterface, SequentialIterator


@SequentialIterator
class Data_Interface(DataInterface):
    def __init__(
        self,
        data_index: Union[list[DataIndex], DataIndex],
        normalisation_params: dict = None,
        transforms: Union[list[TransformCollection], TransformCollection] = None,
        samples: Union[tuple[int], int] = 1,
        sample_interval: int = 0,
        sample_interval_unit: str = "minutes",
    ) -> None:
        """
        An extension of DataIndexes designed for ML Training,
        Using the provided data_index/s allows iteration between bound 
        automatically applying a normalisation and other transforms.

        Also allows for multiple samples to be returned.

        Parameters
        ----------
        data_index
            DataIndex/s to use to retrieve data
        normalisation_params, optional
            Parameters for transform.normalise, as well as which method. 
                If not given, no normalisation, by default None
            Params:
                start - start date for search
                end - end date for search
                interval - interval between searches
                interval_unit - unit of above
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
        sample_interval_unit, optional
            Unit of Above, by default "minutes"

        Raises
        ------
        ValueError
            If Transforms is list for each index but not the right size
        ValueError
            If samples > 1 and sample_interval == 0
        """

        if not isinstance(data_index, (list, tuple)):
            data_index = [data_index]
        self.data_index = data_index

        self.normalisation_params = normalisation_params

        self.transforms = TransformCollection()

        if transforms:
            if transforms == "default":
                transforms = default.get_default_transforms()
            if isinstance(transforms, Iterable) and "default" in transforms:
                transforms.remove("transforms")
                transforms = TransformCollection(transforms)
                transforms += default.get_default_transforms()

        if (
            isinstance(transforms, list)
            and len(transforms) > 0
            and isinstance(transforms, Iterable)
        ):
            if not len(transforms) == len(data_index):
                raise ValueError(
                    f"If transforms is list of seperate TransformCollections, must be same size as data_index. {len(data_index)} != {len(transforms)}"
                )
            self.transforms = transforms
        else:
            self.transforms = [TransformCollection(transforms)] * len(data_index)

        if isinstance(samples, int) and samples > 1 and sample_interval == 0:
            raise ValueError(f"If 'samples' > 1, 'sample_interval' cannot be 0")
        if isinstance(samples, list):
            samples = tuple(samples)

        self.samples = samples
        self.sample_interval = sample_interval
        self.sample_interval_unit = sample_interval_unit

    def set_iterable(
        self,
        start: Union[str, datetime, dset_datetime],
        end: Union[str, datetime, dset_datetime],
        interval: int,
        interval_unit: str = "minutes",
    ):
        """
        Set iteration range for MLDataIndex

        Parameters
        ----------
        start
            Start date of iteration
        end
            End date of iteration
        interval
            Interval between samples
        interval_unit, optional
            Unit of above, by default "minutes"
        """

        self._interval = timedelta(**{interval_unit: interval})
        self._start = dset_datetime(start)
        self._end = dset_datetime(end)

        self._iterator_ready = True

    def rebuild_time(
        self,
        dataset: Union[tuple[xr.Dataset], xr.Dataset],
        time_value: Union[dset_datetime, datetime],
    ):
        if isinstance(dataset, tuple):
            return tuple(map(self.rebuild_time, dataset))

        if not "time" in dataset:
            return dataset

        time_size = len(dataset["time"])
        time_value = dset_datetime(time_value)

        interval = timedelta(**{self.sample_interval_unit: self.sample_interval})

        new_time = [(time_value + interval * i).datetime64() for i in interval]
        return dataset.assign_coords(time=new_time)

    def _retrieve_from_index(
        self,
        timestep: Union[str, datetime, dset_datetime],
        data_index: Union[DataIndex, OperatorIndex],
        transforms: TransformCollection,
    ) -> tuple:
        """
        At given index retrieve given time with samples
        """

        interval = (
            timedelta(**{self.sample_interval_unit: self.sample_interval})
            if isinstance(self.sample_interval, int)
            else self.sample_interval
        )
        timestep = dset_datetime(timestep)

        if self.samples == 1:
            data = data_index.single(timestep, transform=transforms)
            return (data,) if not isinstance(data, tuple) else data

        elif (
            isinstance(self.samples, int)
            and self.samples > 1
            and isinstance(data_index, OperatorIndex)
        ):
            data = data_index.series(
                timestep,
                timestep + interval * (self.samples - 1),
                interval,
                transforms=transforms,
                chunks="auto",
            )
            return (data,) if not isinstance(data, tuple) else data

        elif isinstance(self.samples, tuple):
            if isinstance(data_index, OperatorIndex):
                data_prior = data_index.series(
                    timestep - interval * (self.samples[0] - 1),
                    timestep,
                    interval,
                    transforms=transforms,
                    chunks="auto",
                )
                data_next = data_index.series(
                    timestep + interval,
                    timestep + interval * self.samples[1],
                    interval,
                    transforms=transforms,
                    chunks="auto",
                )
                return data_prior, data_next
            else:
                raise NotImplementedError("TODO: Samples for non OperatorIndex")

        else:
            raise NotImplementedError()

    def _retrieve_at_time(self, timestep: Union[str, datetime, dset_datetime]):
        """
        Retrieve Data from all DataIndexes at given time
        """

        return_list = []
        for i, index in enumerate(self.data_index):
            transforms = TransformCollection(self.transforms[i])
            transforms = transforms + self._normalise[i]

            for element in self._retrieve_from_index(
                timestep,
                index,
                transforms=transforms,
            ):
                return_list.append(element)

        if len(return_list) == 1:
            return return_list[0]
        else:
            return (*tuple(return_list),)

    def __iter__(self):
        if not hasattr(self, "_start"):
            raise RuntimeError(
                f"Iterator not set for {self.__class__.__name__}. Run .set_iterable()"
            )

        steps = (self._end - self._start) / self._interval
        for step in range(int(steps)):
            current_time = self._start + self._interval * step
            yield self._retrieve_at_time(current_time)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.data_index[idx]
        elif isinstance(idx, (str, dset_datetime, datetime)):
            return self._retrieve_at_time(dset_datetime(idx))
        elif isinstance(idx, tuple):
            next_idx = idx[1:]
            if len(next_idx) == 1:
                next_idx = next_idx[0]
            return self[idx[0]].__getitem__(next_idx)
        raise ValueError

    def __call__(self, *idx):
        if isinstance(idx, tuple) and len(idx) == 1:
            idx = idx[0]
        return self[idx]

    def normalise(
        self, dataset: Union[xr.Dataset, tuple[xr.Dataset]], override_index: int = None
    ) -> Union[xr.Dataset, tuple[xr.Dataset]]:
        """
        Apply normalisation like done on retrieval but to given data

        Parameters
        ----------
        dataset
            Dataset/s to apply normalisation to
        override_index, optional
            Override which normalisation is used for single data, by default None

        Returns
        -------
            Normalised Dataset
        """
        if isinstance(dataset, (xr.Dataset, xr.DataArray)):
            return self._normalise[override_index or 0](dataset)
        else:
            return tuple(self._normalise[i](x) for i, x in enumerate(dataset))

    @property
    @functools.lru_cache(None)
    def _normalise(self) -> tuple[Transform]:
        """
        Get normalisation transforms
        """
        if self.normalisation_params is None:
            return [FunctionTransform(lambda x: x)] * len(self.data_index)

        params = dict(self.normalisation_params)
        method = params.pop("method")
        default = params.pop("default", None)

        return tuple(
            normalisation.normalise(index, **params)(method, default)
            for index in self.data_index
        )

    def unnormalise(
        self, dataset: Union[xr.Dataset, tuple[xr.Dataset]], override_index: int = None
    ) -> Union[xr.Dataset, tuple[xr.Dataset]]:
        """
        Apply Unnormalisation to given data.
        Can be used on direct output of self[]

        Parameters
        ----------
        dataset
            Dataset/s to apply unnormalisation to
        override_index, optional
            Override which unnormalisation is used for single data, by default None

        Returns
        -------
            Unnormalised Dataset

        Examples
        --------
            >>> MLIndex.unnormalise(MLIndex[date])
        """
        if isinstance(dataset, (xr.Dataset, xr.DataArray)):
            return self._unnormalise[override_index or 0](dataset)
        else:
            return tuple(self._unnormalise[i](x) for i, x in enumerate(dataset))

    @functools.wraps(unnormalise)
    def undo(self, *args, **kwargs):
        """
        Undo changes done to data
        """
        return self.unnormalise(*args, **kwargs)


    def _unnormalise(self) -> tuple[Transform]:
        """
        Get unnormalisation transforms.
        Is aware of samples sometimes producing multiple datasets
        """
        if isinstance(self.samples, tuple):
            alter = len(self.samples)
        else:
            alter = 1

        if self.normalisation_params is None:
            return [FunctionTransform(lambda x: x)] * len(self.data_index * alter)

        params = dict(self.normalisation_params)
        method = params.pop("method")
        default = params.pop("default", None)

        return tuple(
            normalisation.unnormalise(self.data_index[index // alter], **params)(
                method, default
            )
            for index in range(len(self.data_index) * alter)
        )

    def _formatted_name(self):
        padding = lambda name, length_: name + "".join([" "] * (length_ - len(name)))
        desc = f"Interface for {[index.__class__.__name__ for index in self.data_index]}. Providing {self.samples!r} samples"
        desc = desc.replace("\n", "").replace("\t", "").strip()
        return f"{padding(self.__class__.__name__, 30)}{desc}"

    def __repr__(self) -> str:
        return (
            f"Interface for {[index.__class__.__name__ for index in self.data_index]}"
        )
