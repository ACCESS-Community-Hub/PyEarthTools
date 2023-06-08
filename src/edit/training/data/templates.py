from __future__ import annotations

from abc import abstractmethod
import functools
import logging
import warnings
from typing import Any, Callable, Union

from datetime import datetime

import numpy as np
import xarray as xr

from edit.data import RootIndex, DataIndex, OperatorIndex, Collection
from edit.data.time import EDITDatetime, TimeDelta, time_delta, time_delta_resolution

import edit.training
from edit.training.data.utils import get_pipeline, get_callable



HTML_REPR_ENABLED = False
try:
    import edit.utils
    HTML_REPR_ENABLED = True
except ImportError:
    HTML_REPR_ENABLED = False

RESERVED_NAMES = ['_info_','__doc__']

class DataStep:
    """
    Base Data Pipeline Object
    """

    # def __new__(cls, *args, **kwargs) -> 'Self':
    #     from edit.training.data.sequential import SequentialIterator
    #     print(args)
    #     return SequentialIterator(cls.__init__)(cls, *args, **kwargs)

    def __init__(
        self,
        index: "DataStep",
    ):
        self.index = index
        if HTML_REPR_ENABLED:
            self._repr_html_ = self._repr_html__
            

    @abstractmethod
    def __getitem__(self, idx):
        return self.index[idx]

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()
    
    @property
    def steps(self) -> list[str]:
        """List of steps in Pipeline 

        Returns:
            (list[str]): 
                List of steps in Pipeline 
        """
        step_num = self.step_number
        steps = []
        for num in range(step_num):
            steps.append(self.step(num).__class__.__name__)
        return steps

    def step(self, key: str | type | int | Any) -> "DataStep":
        """Get Step in Pipeline if it matches key

        Args:
            key (str | type | int | Any):
                Key for step to be retrieved

        Returns:
            (DataStep):
                Step in pipeline matching key
        """
        if isinstance(key, str) and self.__class__.__name__ == key:
            return self
        elif isinstance(key, type) and isinstance(self, key):
            return self
        elif isinstance(key, int):
            if key < 0:
                key = self.step_number - abs(key + 1)
            if key == self.step_number:
                return self
        elif key == self:
            return self

        try:
            if isinstance(self.index, (DataInterface, DataStep)):
                return self.index.step(key)
        except KeyError:
            pass
        if isinstance(key, int):
            raise KeyError(f"Could not find pipeline step {key!r} in DataPipeline. Steps length is {self.step_number}")
        raise KeyError(f"Could not find {key!r} in Data Pipeline")

    @property
    # @functools.lru_cache(1)
    def step_number(self):
        # print(isinstance(self.index, DataStep), type(self.index))
        if isinstance(self.index, DataStep):
            return self.index.step_number + 1
        return 0

    def __getattr__(self, key):
        if key == "index" or key in RESERVED_NAMES:
            raise AttributeError(f"{self.__class__} has no attribute {key!r}")
        try:
            return getattr(self.index, key)
        except AttributeError as e:
            pass
        raise AttributeError(f"DataPipeline has no attribute {key!r}")

    def __call__(self, idx):
        return self.__getitem__(idx)

    """
    repr's
    """

    @property
    def _formatted_doc_(self):
        desc = self.__doc__ or "No Docstring"
        desc_list = desc.strip().split("\n")
        if "" in desc_list:
            desc_list.remove("")
        return desc_list[0].replace('\t','').strip()

        
    def _get_steps_for_repr_(self):
        pipeline_steps = [self.step(num) for num in range(0, self.step_number + 1)]

        class formatting_wrapper:
            def __init__(self, object):
                self.object = object

            @property
            def __class__(self):
                return self.object.__class__

            @property
            def _formatted_doc_(self):
                if isinstance(self.object, (list, tuple)):
                    return f"List "# containing {[obj.__class__.__name__ for obj in self.object]}"
                desc = self.object.__doc__ or "No Docstring"
                desc_list = desc.strip().split("\n")
                if "" in desc_list:
                    desc_list.remove("")
                return desc_list[0].replace('\t','').strip()

            @property
            def _info_(self):
                info_dict = {}
                if isinstance(self.object, (list, tuple)):
                    for i, obj in enumerate(self.object):
                        if hasattr(obj, 'variables'):
                            info_dict.update({f"{obj.__class__.__name__}{i}": obj.variables})
                            continue
                        else:
                            info_dict.update({f"{obj.__class__.__name__}{i}": obj.__doc__})
                
                return info_dict

        if not isinstance(self.step(0).index, DataStep):
            pipeline_steps = [formatting_wrapper(self.step(0).index), *pipeline_steps]
        return pipeline_steps

    def _repr_html__(self) -> str:
        if not HTML_REPR_ENABLED:
            raise KeyError(f"{self!r} has no attribute '_repr_html_'")
        pipeline_steps = self._get_steps_for_repr_()

        return edit.utils.repr.html(*pipeline_steps, name = 'Data', documentation_attr = '_formatted_doc_', info_attr = '_info_', backup_repr = self.__repr__())

    def __repr__(self):
        string = "Data Pipeline of the following:\n"
        padding = lambda name, length_: name + "".join([" "] * (length_ - len(name)))

        pipeline_steps = self._get_steps_for_repr_()
        return edit.utils.repr.standard(*pipeline_steps, name = 'Data', documentation_attr = '_formatted_doc_', info_attr = '_info_')

        # for step in pipeline_steps:
        #     formatted = f"\t* {padding(step.__class__.__name__, 25)}{step._formatted_doc_}\n"
        #     if hasattr(step, '_info_'):
        #         formatted = formatted + f"\t\t\t\t\t{step._info_}\n"
        #     string += formatted #+ '\n'

        # return string
    
    def __str__(self):
        return f"DataPipeline containing {self.steps}"
    

    def plot(self, idx = None, **kwargs):
        return edit.training.data.sanity.plot(self, index = idx, **kwargs)

    def summary(self, idx = None, **kwargs):
        return edit.training.data.sanity.summary(self, idx, **kwargs)

    @property
    def ignore_sanity(self):
        return False

class DataOperation(DataStep):
    """
    Base DataOperation. A pipeline step with which an operation can be applied to the data.
    """

    def __init__(
        self,
        index : DataStep | DataOperation,
        apply_func: Callable,
        undo_func: Callable,
        *,
        apply_iterator: bool = True,
        apply_get: bool = True,
        split_tuples: bool = False,
        recognised_types: tuple[type] = None,
        doc: str = None,
    ) -> None:
        """Base DataOperation, 

        Applies given functions on given steps

        Args:
            index (DataStep): 
                Underlying pipeline step with which to get data from
            apply_func (Callable): 
                Function to apply to data. Can be None to not apply.
            undo_func (Callable): 
                Function to apply to data to `undo` the `apply_func`. Can be None to not undo.
            apply_iterator (bool, optional): 
                Apply on iteration. Defaults to True.
            apply_get (bool, optional): 
                Apply on __getitem__. Defaults to True.
            split_tuples (bool, optional):
                Split tuples of data, applying the given functions to each element. Defaults to False.
            recognised_types (tuple[type], optional):
                List or tuple of recognised types, will raise an exception if data is passed of the wrong type. Defaults to None.
            doc (str, optional):
                Override for __doc__ string. Defaults to None.

        Examples:
            ## Applying a Function
            >>> operation = DataOperation(DataStep, apply_func = lambda x: x + 1, undo_func = lambda x: x - 1)
            >>> operation.apply_func(0) # Apply this operation to a given data
            1
            ## Undoing a Function
            >>> operation.undo_func(1) # Apply the undo operation to a given data
            0
            ## Indexing Data
            >>> operation = DataOperation([41], apply_func = lambda x: x + 1, undo_func = lambda x: x - 1)
            >>> operation[0]
            42  # Data from DataStep at [0] with lambda x: x + 1 applied
            ## Iterating Through Data
            >>> list(operation)
            [42]
            >>> operation.undo(list(operation))
            41

        """        
        super().__init__(index)

        self._apply_func = apply_func
        self._undo_func = undo_func

        self.split_tuples = split_tuples


        if not recognised_types is None:
            if isinstance(recognised_types, list):
                recognised_types = tuple(recognised_types)
            elif not isinstance(recognised_types, tuple):
                recognised_types = (recognised_types,)
            if self.split_tuples:
                recognised_types = (tuple, list, *recognised_types)
        self.recognised_types = recognised_types

        self.apply_iterator = apply_iterator
        self.apply_get = apply_get

        if doc:
            self.__doc__ = doc


    def check_types(self, data: Any) -> bool:
        if self.recognised_types is None:
            return True
        if not isinstance(data, self.recognised_types):
            raise TypeError(f"{self.__class__.__name__} cannot handle '{type(data)}'. Recognised types are: {self.recognised_types}")
        if isinstance(data, (tuple, list)):
            if not isinstance(data[0], self.recognised_types):
                raise TypeError(f"{self.__class__.__name__} cannot handle an iterable containing '{type(data[0])}'. Recognised types are: {self.recognised_types}")
        return True
        
    def apply_func(self, data: xr.Dataset | xr.DataArray | np.ndarray | tuple) -> xr.Dataset | xr.DataArray | np.ndarray | tuple:
        """Apply the given `apply_func` from init.

        Will automatically split / join tuples and check types if given by init arguments

        Args:
            data (xr.Dataset | xr.DataArray | np.ndarray | tuple): 
                Data to apply `apply_func` to

        Returns:
            (xr.Dataset | xr.DataArray | np.ndarray | tuple): 
                Data with `apply_func` function applied to it
        """        
        self.check_types(data)

        try:
            if isinstance(data, (list, tuple)) and self.split_tuples:
                return tuple(map(self.apply_func, data))
        except NotImplementedError:
            pass
        if self._apply_func:
            return self._apply_func(data)
        return data

    def undo_func(self, data: xr.Dataset | xr.DataArray | np.ndarray | tuple) -> xr.Dataset | xr.DataArray | np.ndarray | tuple:
        """Apply the given `undo_func` from init.

        Will automatically split / join tuples and check types if given by init arguments

        Args:
            data (xr.Dataset | xr.DataArray | np.ndarray | tuple): 
                Data to apply `undo_func` to

        Returns:
            (xr.Dataset | xr.DataArray | np.ndarray | tuple): 
                Data with `undo_func` function applied to it
        """            
        self.check_types(data)
        try:
            if isinstance(data, (list, tuple)) and self.split_tuples:
                return tuple(map(self.undo_func, data))
        except NotImplementedError:
            pass

        if self._undo_func:
            return self._undo_func(data)
        return data

    def undo(self, data: xr.Dataset | xr.DataArray | np.ndarray | tuple) -> xr.Dataset | xr.DataArray | np.ndarray | tuple:
        """Undo transforms and edits the Data Pipeline has done

        Args:
            data (xr.Dataset | xr.DataArray | np.ndarray | tuple):
                Data from this pipeline to undo changes from

        Returns:
            (xr.Dataset | xr.DataArray | np.ndarray | tuple):
                Result of `.undo` from the Pipeline
        """
        data = self.undo_func(data)
        
        if hasattr(self.index, "undo"):
            data = self.index.undo(data)
        return data

    def apply(self, data: xr.Dataset | xr.DataArray | np.ndarray | tuple):
        """Apply DataPipeline to given Data

        Args:
            data (xr.Dataset | xr.DataArray | np.ndarray):
                Data to apply pipeline to

        Returns:
            (xr.Dataset | xr.DataArray | np.ndarray | tuple):
                Result of Data Pipeline steps
        """

        if hasattr(self.index, "apply"):
            data = self.index.apply(data)
        return self.apply_func(data)

    def get(self, *args, **kwargs):
        if hasattr(self.index, "get"):
            data = self.index.get(*args, **kwargs)
            if self.apply_get:
                data = self.apply_func(data)
            return data
        raise AttributeError(f"{self} has no attribute 'get'")

    def __iter__(self):
        for data in self.index:
            if self.apply_iterator:
                yield self.apply_func(data)
            else:
                yield data

    def __getitem__(self, idx):
        if self.apply_get:
            data = self.index[idx]
            return self.apply_func(data)
        return self.index[idx]

    def __call__(self, idx):
        if isinstance(idx, str | EDITDatetime):
            return self.__getitem__(idx)
        else:
            return self.apply(idx)

class TrainingRootIndex(RootIndex, DataStep):
    """
    [edit.data.DataIndex][edit.data.DataIndex] as a Pipeline step
    """
    # def __new__(cls, *args, **kwargs) -> 'Self':
    #     from edit.training.data.sequential import SequentialIterator
    #     print('cls', type(cls))
    #     return SequentialIterator(cls.__init__)(cls, *args, **kwargs)

    def __init__(
        self,
        index: list[TrainingRootIndex] | TrainingRootIndex | DataIndex,
        *,
        allow_multiple_index: bool = False,
        **kwargs,
    ) -> None:
        """Combine an DataIndex as a DataStep in a pipeline

        Args:
            index (list[TrainingDataIndex] | TrainingDataIndex | DataIndex): 
                Underlying DataIndex to use to get data
            allow_multiple_index (bool, optional): 
                Allow multiple indexes to be set. Defaults to False.
        """    
        if isinstance(index, dict):
            index = get_pipeline(index)
        if not allow_multiple_index and isinstance(index, (list, tuple)):
            index = index[0]
        elif allow_multiple_index and not isinstance(index, (tuple, list)):
            index = (index,)
        
        if allow_multiple_index:
            index = Collection(*index)
        self.index = index

        super().__init__(add_default_transforms = kwargs.pop('add_default_transforms', False), **kwargs)

        if HTML_REPR_ENABLED:
            self._repr_html_ = self._repr_html__


    def __getattr__(self, key):
        if key == "index":
            raise AttributeError(f"{self.__class__} has no attribute {key}")
        index = self.index
        return getattr(index, key)

    def undo(self, data, *args, **kwargs):
        if hasattr(self.index, "undo"):
            return self.index.undo(data, *args, **kwargs)
        return data

class TrainingOperatorIndex(OperatorIndex, DataStep):
    """
    [edit.data.OperatorIndex][edit.data.OperatorIndex] as a Pipeline step
    """

    def __init__(
        self,
        index: list[TrainingOperatorIndex] | TrainingOperatorIndex | OperatorIndex,
        *,
        allow_multiple_index: bool = False,
        **kwargs,
    ) -> None:
        """Combine an OperatorIndex as a DataStep in a pipeline

        Args:
            index (list[TrainingOperatorIndex] | TrainingOperatorIndex | OperatorIndex): 
                Underlying OperatorIndex to use to get data
            allow_multiple_index (bool, optional): 
                Allow multiple indexes to be set. Defaults to False.
        """    
        if isinstance(index, dict):
            index = get_pipeline(index)
        if not allow_multiple_index and isinstance(index, (list, tuple)):
            index = index[0]
        elif allow_multiple_index and not isinstance(index, (tuple, list)):
            index = (index,)
        if allow_multiple_index:
            index = Collection(*index)
        self.index = index

        if "data_resolution" not in kwargs and not allow_multiple_index:
            kwargs["data_resolution"] = index.data_interval
        super().__init__(add_default_transforms = kwargs.pop('add_default_transforms', False), **kwargs)
        
        if HTML_REPR_ENABLED:
            self._repr_html_ = self._repr_html__

    def __getattr__(self, key):
        if key == "index":
            raise AttributeError(f"{self.__class__} has no attribute {key}")
        index = self.index
        return getattr(index, key)

    def undo(self, data, *args, **kwargs):
        if hasattr(self.index, "undo"):
            return self.index.undo(data, *args, **kwargs)
        return data

class DataInterface(DataOperation, OperatorIndex):
    """
    Training DataOperation that requires a DataIndex underneath it

    Allows OperatorIndex calls as well as DataOperation calls
    Usually sits between OperatorIndexes and DataOperation's
    """

    def __init__(self, index: OperatorIndex, **datastep_kwargs) -> None:
        """An `OperatorIndex` as a [DataOperation][edit.training.data.templates.DataOperation]

        Args:
            index (OperatorIndex): 
                Underlying OperatorIndex to use to get data
        """        
        
        super().__init__(index=index, **datastep_kwargs)

    def get(self, querytime: str | EDITDatetime):
        return self.index[querytime]

class DataIterator(DataStep):
    """
    Provide a way to iterator over data, and catch known errors.

    A child of this class must end the indexes & interfaces section of the data loader.
    """

    def __init__(
        self,
        index: DataStep,
        catch: tuple[Exception] | tuple[str] | Exception | str = None,
        warnings: tuple[warning] | tuple[str] | warning | str = None,
    ) -> None:
        """Iterate over Data between date ranges

        Args:
            index (DataStep): 
                Underlying DataStep to get data from
            catch (tuple[Exception] | Exception, optional): 
                Errors to catch, either defined or names of. Defaults to None.
        """    
        super().__init__(index)
        
        def get_callables(callables):
            callables = [callables] if not isinstance(catch, (tuple, list)) else list(catch)
            
            for i, call in enumerate(callables):
                if isinstance(call, str):
                    callables[i] = get_callable(call)
            return callables
        
        catch = get_callables(catch) if catch else []
        warnings = get_callables(warnings) if warnings else []
        
        self._error_to_catch: tuple[Exception] = tuple(catch)
        self._warnings_to_catch = tuple(warnings)
        
        self._info_ = dict(NotConfigured = True)
        self._iterator_ready = False

    def __getitem__(self, idx: str):
        return self.index[idx]

    def set_iterable(
        self,
        start: str | datetime | EDITDatetime,
        end: str | datetime | EDITDatetime,
        interval: int | tuple,
        *,
        resolution: str = None
    ):
        """Set iteration range for DataIterator

        Args:
            start (str | datetime | EDITDatetime): 
                Start date of iteration
            end (str | datetime | EDITDatetime): 
                End date of iteration
            interval (int | tuple): 
                Interval between samples
                Use [pandas.to_timedelta][pandas.to_timedelta] notation, (10, 'minute')
            resolution (str, optional):
                Override for resolution
        """   

        self._interval = TimeDelta(interval)

        self._start: EDITDatetime = EDITDatetime(start).at_resolution(resolution or self._interval.resolution)
        self._end: EDITDatetime = EDITDatetime(end)#.at_resolution(self._interval)

        self._iterator_ready = True
        self._info_.update(dict(start = self._start, end = self._end, interval = self._interval))
        
        if 'NotConfigured' in self._info_:
            self._info_.pop('NotConfigured')

        if hasattr(self.index, 'set_iterable'):
            self.index.set_iterable(start, end, interval)

    def __iter__(self):
        if not self._iterator_ready:
            raise RuntimeError(
                f"Iterator not set for {self.__class__.__name__}. Run .set_iterable()"
            )

        steps = (self._end - self._start) // self._interval
        tuple(warnings.filterwarnings(action = 'once', category = warn) for warn in self._warnings_to_catch)
        
        for step in range(int(steps)):
            try:
                current_time = self._start + (self._interval * step)
                yield self[current_time]
            except self._error_to_catch as e:
                logging.info(e)

    def ignore_sanity(self):
        return 'Iterator'