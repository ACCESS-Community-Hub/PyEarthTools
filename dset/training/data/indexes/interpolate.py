
from typing import Any, Union
import xarray as xr

import dset.data
from dset.data import archive, transform, TransformCollection, DSETDatetime
from dset.data.default import OperatorIndex

from dset.training.data.utils import get_transforms
from dset.training.data.templates import SequentialIterator, TrainingOperatorIndex

@SequentialIterator
class InterpolationIndex(TrainingOperatorIndex):
    """
    General OperatorIndex capable of taking any other defined OperatorIndex and combining them.
    """
    def __init__(self, 
        indexes: Union[list,dict, OperatorIndex],
        sample_interval: tuple[int, tuple[int]] = None,
        transforms : Union[list, dict] = TransformCollection(),
        interpolation: Any = None, 
        interpolation_method: str = 'linear'
        ):
        """
        GeneralIndexer which interpolates all given indexes together.

        Will retrieve samples with sample_interval resolution.

        Parameters
        ----------
        indexes
            Dictionary with keys as imports or modules to other OperatorIndexes
        sample_interval, optional
            Sample Interval to pass up, must be of pandas.to_timestep form.
            E.g. (10,'H') - 10 Hours
        transforms, optional
            Other Transforms to apply, by default []
        interpolation, optional
            Reference interpolation dataset, if not given use first dataset, by default None
        interpolation_method, optional
            Interpolation Method, by default 'linear'
        """


        self.interpolation = interpolation
        self.interpolation_method = interpolation_method

        if isinstance(transforms, dict):
            transforms = get_transforms(transforms)

        base_transforms = TransformCollection(transforms) 
        
        super().__init__(indexes, base_transforms, sample_interval, allow_multiple_index=True)


    def get(self, query_time, **kwargs):
        data = []

        for index in self.index:
            new_data = index(query_time, transforms = self.base_transforms, **kwargs)
            if data:
                interp = transform.interpolation(self.interpolation or data[-1], method = self.interpolation_method, drop_coords = 'time')
                if 'time' in new_data.indexes:
                    new_data = new_data.sel(time = DSETDatetime(query_time).datetime64())
                    new_data = interp(new_data)
                elif 'time' not in new_data:
                    try:
                        new_data = new_data.assign_coords(time = data[-1].time) #[DSETDatetime(query_time).datetime64()]
                    except ValueError:
                        new_data = new_data.assign_coords(time = [DSETDatetime(query_time).datetime64()])
                    new_data = interp(new_data)
                else:
                    pass
                    #new_data['time'] = data[-1]['time']
            data.append(new_data)
        ds = xr.merge(data)
        return ds

    def __repr__(self):
        return_string = "General Index Combining: \n"
        for index in self.index:
            return_string += f"\t{index}\n"
        return return_string

    def _formatted_name(self):
        padding = lambda name, length_: name + "".join([" "] * (length_ - len(name)))
        desc = f"Interpolation Index for {[index.__class__.__name__ for index in self.index]!r}. {self.interpolation_method} interpolating all together"
        desc = desc.replace("\n", "").replace("\t", "").strip()
        formatted = f"{padding(self.__class__.__name__, 30)}{desc}"

        for index in self.index:
            if hasattr(index, '_formatted_name'):
                formatted += f"\n{index._formatted_name()}"
        return formatted