from __future__ import annotations

from abc import abstractmethod
import logging
from typing import Any, Callable, Union

from datetime import datetime


from edit.data import DataIndex, OperatorIndex
from edit.data.time import EDITDatetime, time_delta, time_delta_resolution
from edit.training.data.utils import get_pipeline, get_callable

HTML_REPR_ENABLED = False
try:
    import edit.utils
    HTML_REPR_ENABLED = True
except ImportError:
    HTML_REPR_ENABLED = False

class DataStep:
    """
    Base Data Pipeline Object
    """

    def __init__(
        self,
        index: "DataStep",
    ):
        self.index = index

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError()

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()

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
        elif isinstance(key, int) and key == self.step_number or key == -1:
            return self
        elif key == self:
            return self

        if isinstance(self.index, DataStep):
            return self.index.step(key)
        else:
            raise KeyError(f"Could not find {key!r} in Data Pipeline")

    @property
    def step_number(self):
        if isinstance(self.index, DataStep):
            return self.index.step_number + 1
        else:
            return 0

    def __getattr__(self, key):
        if key == "index":
            raise AttributeError(f"{self.__class__} has no attribute {key!r}")
        return getattr(self.index, key)

    def __call__(self, idx):
        if isinstance(idx, str | EDITDatetime):
            return self.__getitem__(idx)
        else:
            return self.apply(idx)

    """
    repr's
    """

    @property
    def _formatted_doc_(self):
        desc = self.__doc__ or "No Docstring"
        desc_list = desc.strip().split("\n")
        if "" in desc_list:
            desc_list.remove("")
        desc = desc_list[0].replace('\t','').strip()
        return desc

        
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
                    return f"List containing: {[obj.__class__.__name__ for obj in self.object]}"
                desc = self.object.__doc__ or "No Docstring"
                desc_list = desc.strip().split("\n")
                if "" in desc_list:
                    desc_list.remove("")
                desc = desc_list[0].replace('\t','').strip()
                return desc

        if not isinstance(self.step(0).index, DataStep):
            pipeline_steps = [formatting_wrapper(self.step(0).index), *pipeline_steps]
        return pipeline_steps

    def _repr_html_(self) -> str:
        if not HTML_REPR_ENABLED:
            raise AttributeError(f"{self!r} has no attribute '_repr_html_'")
        pipeline_steps = self._get_steps_for_repr_()

        return edit.utils.repr.provide_html(*pipeline_steps, name = 'Data Pipeline', documentation_attr = '_formatted_doc_', backup_repr = self.__repr__())

    def __repr__(self):
        string = "Data Pipeline of the following:\n"
        padding = lambda name, length_: name + "".join([" "] * (length_ - len(name)))

        pipeline_steps = self._get_steps_for_repr_()

        for step in pipeline_steps:
            formatted = f"\t*{padding(step.__class__.__name__, 30)}{step._formatted_doc_}\n"
            string += formatted

        return string


    @property
    def ignore_sanity(self):
        return False


class DataOperation(DataStep):
    """
    Base DataOperation. A pipeline step with which an operation can be applied to the data.

    """

    def __init__(
        self,
        index : DataStep,
        apply_func: Callable,
        undo_func: Callable,
        *,
        apply_iterator: bool = True,
        apply_get: bool = True,
    ) -> None:
        """Base DataOperation, 

        Applies given functions on given steps

        Args:
            index (DataStep): 
                Underlying pipeline step with which to get data from
            apply_func (Callable): 
                Function to apply to data
            undo_func (Callable): 
                Function to apply to data to `undo` the `apply_func`
            apply_iterator (bool, optional): 
                Apply on iteration. Defaults to True.
            apply_get (bool, optional): 
                Apply on __getitem__. Defaults to True.
        """        
        super().__init__(index)

        self.apply_func = apply_func
        self.undo_func = undo_func

        self.apply_iterator = apply_iterator
        self.apply_get = apply_get

    def undo(self, data):
        """Undo transforms and edits the Data Pipeline has done

        Args:
            data (xr.Dataset | xr.DataArray | np.ndarray):
                Data from this pipeline to undo changes from

        Returns:
            (xr.Dataset | xr.DataArray | np.ndarray):
                Result of `.undo` from the Pipeline
        """
        if self.undo_func is not None:
            data = self.undo_func(data)
        if hasattr(self.index, "undo"):
            data = self.index.undo(data)
        return data

    def apply(self, data):
        """Apply DataPipeline to given Data

        Args:
            data (xr.Dataset | xr.DataArray | np.ndarray):
                Data to apply pipeline to

        Returns:
            (xr.Dataset | xr.DataArray | np.ndarray):
                Result of Data Pipeline steps
        """
        if hasattr(self.index, "apply"):
            data = self.index.apply(data)
        if self.apply_func is not None:
            data = self.apply_func(data)
        return data

    def __iter__(self):
        for data in self.index:
            if self.apply_iterator and self.apply_func:
                yield self.apply_func(data)
            else:
                yield data

    def __getitem__(self, idx):
        if self.apply_get and self.apply_func:
            return self.apply_func(self.index[idx])
        return self.index[idx]


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
        self.index = index

        if "data_resolution" not in kwargs and not allow_multiple_index:
            kwargs["data_resolution"] = index.data_resolution
        super().__init__(**kwargs)

    def __getattr__(self, key):
        if key == "index":
            raise AttributeError(f"{self.__class__} has no attribute {key}")
        index = self.index
        if isinstance(self.index, (list, tuple)):
            index = self.index[0]
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
    ) -> None:
        """Iterate over Data between date ranges

        Args:
            index (DataStep): 
                Underlying DataStep to get data from
            catch (tuple[Exception] | Exception, optional): 
                Errors to catch, either defined or names of. Defaults to None.
        """    
        super().__init__(index)

        if catch:
            catch = [catch] if not isinstance(catch, (tuple, list)) else catch

            for i, err in enumerate(catch):
                if isinstance(err, str):
                    catch[i] = get_callable(err)
        else:
            catch = []
        self._error_to_catch: tuple[Exception] = tuple(catch)

    def __getitem__(self, idx: str):
        return self.index[idx]

    def set_iterable(
        self,
        start: str | datetime | EDITDatetime,
        end: str | datetime | EDITDatetime,
        interval: int | tuple,
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
        """   

        self._interval = time_delta(interval)

        self._start = EDITDatetime(start).at_resolution(self._interval)
        self._end = EDITDatetime(end).at_resolution(self._interval)

        self._iterator_ready = True

    def __iter__(self):
        if not hasattr(self, "_start"):
            raise RuntimeError(
                f"Iterator not set for {self.__class__.__name__}. Run .set_iterable()"
            )

        steps = (self._end - self._start) // self._interval

        for step in range(int(steps)):
            try:
                current_time = self._start + (self._interval * step)
                yield self[current_time]
            except self._error_to_catch as e:
                logging.info(e)
