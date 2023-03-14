
from typing import Any, Union
import xarray as xr
import datetime

import importlib

import dset.data
from dset.data import archive, transform, TransformCollection, dset_datetime
from dset.data.default import OperatorIndex


def get_callable(module: str) -> "DataIterator":
    """
    Provide dynamic import capability

    Parameters
    ----------
        module
            String of path the module, either module or specific function/class

    Returns
    -------
        Specified module or function
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError:
        module = module.split(".")
        return getattr(get_callable(".".join(module[:-1])), module[-1])
    
def get_class(root_module, class_name):
    if not class_name:
        return root_module
    if isinstance(class_name, str):
        class_name = class_name.split('.')
    return get_class(getattr(root_module, class_name[0]), class_name[1:])


def get_indexes(sources: dict):
    indexes = []
    
    for index, kwargs in sources.items():
        data_index = None
        try:
            data_index = get_class(dset.data, index)
        except:
            pass
        
        if not data_index:
            for alterations in ["", "dset.data.", "__main__."]:
                try:
                    data_index = get_callable(alterations + index)
                except (ModuleNotFoundError, ImportError, AttributeError, ValueError):
                    pass
                if data_index:
                    break
                
        if not data_index:
            raise ValueError(f"Unable to load {index!r}")

        if not callable(data_index):
            if hasattr(data_index, index.split(".")[-1]):
                data_index = getattr(data_index, index.split(".")[-1])
            else:
                raise TypeError(
                    f"{index!r} is a {type(data_index)}, must be callable"
                )

        indexes.append(data_index(**kwargs))
    return indexes
            
class GeneralIndexer(OperatorIndex):
    """
    General OperatorIndex capable of taking any other defined OperatorIndex and combining them.
    """
    def __init__(self, 
        sources: dict,
        sample_interval: tuple[int, tuple[int]] = (10, 'm'),
        location: Any = None,
        transformers = [],
        interpolation: Any = None, 
        interpolation_method: str = 'linear'
        ):
        """
        Create GeneralIndexer

        Parameters
        ----------
        sources
            Dictionary with keys as imports or modules to other OperatorIndexes
        sample_interval, optional
            Sample Interval to pass up, must be of pandas.to_timestep form.
            E.g. (10,'H') - 10 Hours
            by default (10, m)
        location, optional
            Location Transform, by default None
        transformers, optional
            Other Transforms to apply, by default []
        interpolation, optional
            Reference interpolation dataset, by default None
        interpolation_method, optional
            Interpolation Method, by default 'linear'
        """
        self.indexes = get_indexes(sources)
        self.interpolation = interpolation
        self.interpolation_method = interpolation_method

        base_transform = TransformCollection(transformers) 
        if location:
            base_transform += transform.region(location) 
        super().__init__(base_transform, sample_interval)
        
    def get(self, query_time):
        data = []
        for index in self.indexes:
            per_index_transforms = self.base_transforms
            
            if data:
                interp = transform.Interpolate(self.interpolation or data[-1], method = self.interpolation_method, drop_coords = 'time')
                per_index_transforms += interp
            new_data = index(query_time, transforms = per_index_transforms)
            if data:
                new_data['time'] = data[-1]['time']
            data.append(new_data)
        
        ds = xr.merge(data, combine_attrs = 'drop_conflicts')
        ds = ds.assign_coords(time = [dset_datetime(query_time).datetime64()])
        return ds