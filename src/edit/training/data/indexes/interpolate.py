from typing import Any, Callable, Union
import xarray as xr

from edit.data import transform, TransformCollection, EDITDatetime, operations
from edit.data import OperatorIndex

from edit.training.data.utils import get_transforms
from edit.training.data.templates import TrainingOperatorIndex
from edit.training.data.sequential import SequentialIterator


@SequentialIterator
class InterpolationIndex(TrainingOperatorIndex):
    """
    OperatorIndex which interpolates and returns data from any given indexes on the same spatial grid


    !!! Example
        ```python
        InterpolationIndex(PipelineStep, interpolation_method = 'linear')

        ## As this is decorated with @SequentialIterator, it can be partially initialised

        partialInterpolation = InterpolationIndex(interpolation_method = 'linear')
        partialInterpolation(PipelineStep)
        ```
    """

    def __init__(
        self,
        indexes: list | dict | OperatorIndex,
        sample_interval: tuple[int, tuple[int]] = None,
        transforms: list | dict = TransformCollection(),
        interpolation: Any = None,
        interpolation_method: str = "linear",
        temporal: bool = False,
        temporal_reference: xr.Dataset = None,
        temporal_function: str | Callable = "mean",
    ):
        """OperatorIndex which interpolates any given indexes onto the same spatial grid

        Will retrieve samples with `sample_interval` resolution.

        Args:
            indexes (list | dict | OperatorIndex):
                Indexes in which to interpolate together and return, can be fully defined or dictionary defined
            sample_interval (tuple[int, tuple[int]], optional):
                Sample Interval to pass up, must be of pandas.to_timestep form.
                E.g. (10,'H') - 10 Hours. Defaults to None.
            transforms (list | dict, optional):
                 Other Transforms to apply. Defaults to TransformCollection().
            interpolation (Any, optional):
                Reference Spatial interpolation dataset, if not given use first dataset. Defaults to None.
            interpolation_method (str, optional):
                Interpolation Method, must be in [xarray interp][xarray.Dataset.interp]. Defaults to "linear".
            temporal (bool, optional):
                Temporally Interpolate Datasets together. Defaults to False.
            temporal_reference (xr.Dataset, optional):
                Reference Temporal interpolation dataset, if not given use first dataset. Defaults to None.
            temporal_function (str | Callable, optional):
                Function to use for temporal interpolation. Defaults to 'mean'.
        """

        self.interpolation = interpolation
        self.interpolation_method = interpolation_method

        self.temporal = temporal
        self.temporal_reference = temporal_reference
        self.temporal_function = temporal_function

        if isinstance(transforms, dict):
            transforms = get_transforms(transforms)

        base_transforms = TransformCollection(transforms)

        super().__init__(
            indexes,
            base_transforms=base_transforms,
            data_resolution=sample_interval,
            allow_multiple_index=True,
        )

    def get(self, query_time, **kwargs) -> xr.Dataset:
        """
        Get Data at given time from all given indexes, and interpolate as defined.

        Args:
            query_time (Any):
                Time to retrieve data at

        Returns:
            (xr.Dataset):
                [xr.Dataset][xarray.Dataset] containing data from all indexes interpolated together
        """
        data = []

        for index in self.index:
            new_data = index(query_time, transforms=self.base_transforms, **kwargs)
            if data:
                interp = transform.interpolation(
                    self.interpolation or data[-1],
                    method=self.interpolation_method,
                    drop_coords="time",
                )
                if "time" in new_data.indexes:
                    new_data = new_data.sel(time=EDITDatetime(query_time).datetime64())
                    new_data = interp(new_data)
                elif "time" not in new_data:
                    try:
                        new_data = new_data.assign_coords(
                            time=data[-1].time
                        )  # [EDITDatetime(query_time).datetime64()]
                    except ValueError:
                        new_data = new_data.assign_coords(
                            time=[EDITDatetime(query_time).datetime64()]
                        )
                    new_data = interp(new_data)
                else:
                    pass
                    # new_data['time'] = data[-1]['time']
            data.append(new_data)

        if self.temporal:
            return operations.interpolation.TemporalInterpolation(
                *data,
                reference_dataset=self.temporal_reference,
                aggregation_function=self.temporal_function,
                merge=True,
            )
        return xr.merge(data)

    @property
    def __doc__(self):
        return f"Interpolation Index for {[index.__class__.__name__ for index in self.index]!r}. Uses {self.interpolation_method}."
