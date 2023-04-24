import functools
from abc import abstractmethod
from typing import Callable, Union

import yaml
import inspect
from datetime import datetime



from dset.data import DataIndex, OperatorIndex
from dset.data.time import DSETDatetime, time_delta
from dset.training.data.utils import get_indexes, get_callable


class DataStep():
    """
    A step between the dset.data.DataIndex's and the training pipeline
    """
    def __init__(self, index: 'DataStep | DataIterator') -> None:
        self.index = index

    def __getitem__(self, idx):
        return self.index[idx]

    def undo(self, data, *args, **kwargs):
        if hasattr(self.index, 'undo'):
            return self.index.undo(data, *args, **kwargs)
        return data
        
    @abstractmethod
    def __iter__(self):
        raise NotImplementedError("Data step does not implement '__iter__', child must.")

    def __getattr__(self, key):
        if key == "index":
            raise AttributeError(f"{self.__class__} has no attribute {key}")
        return getattr(self.index, key)

    def __call__(self, idx):
        return self.__getitem__(idx)

    def __repr__(self):
        string = "Data Pipeline with the following:"
        operations = self._formatted_name()
        operations = operations.split("\n")
        operations.reverse()
        operations = "\n".join(["\t* " + oper for oper in operations])
        return f"{string}\n{operations}"

    def _formatted_name(self, desc: str = None):
        padding = lambda name, length_: name + "".join([" "] * (length_ - len(name)))
        desc = desc or self.__doc__ or "No Docstring"
        desc = desc.replace("\n", "").replace("\t", "").strip()
        formatted = f"{padding(self.__class__.__name__, 30)}{desc}"

        if hasattr(self.index, '_formatted_name'):
            formatted += f"\n{self.index._formatted_name()}"
        return formatted
        
    @property
    def ignore_sanity(self):
        return False

class TrainingOperatorIndex(OperatorIndex):
    """
    DSET Training Version of dset.data.OperatorIndex
    """
    def __init__(self, index: "list[TrainingOperatorIndex] | TrainingOperatorIndex", *args, allow_multiple_index: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if isinstance(index, dict):
            index = get_indexes(index)
        if not allow_multiple_index and isinstance(index, (list, tuple)):
            index = index[0]
        self.index = index

    def __getattr__(self, key):
        if key == "index":
            raise AttributeError(f"{self.__class__} has no attribute {key}")
        index = self.index
        if isinstance(self.index, (list, tuple)):
            index = self.index[0]
        return getattr(index, key)

    def undo(self, data, *args, **kwargs):
        if hasattr(self.index, 'undo'):
            return self.index.undo(data, *args, **kwargs)
        return data

    def __repr__(self):
        string = "Training Index: \n"
        operations = self._formatted_name()
        operations = operations.split("\n")
        operations.reverse()
        operations = "\n".join(["\t* " + oper for oper in operations])
        return f"{string}\n{operations}"

    def _formatted_name(self, desc: str = None):
        padding = lambda name, length_: name + "".join([" "] * (length_ - len(name)))
        desc = desc or self.__doc__ or "No Docstring"
        desc = desc.replace("\n", "").replace("\t", "").strip()
        formatted = f"{padding(self.__class__.__name__, 30)}{desc}"

        if hasattr(self.index, '_formatted_name'):
            formatted += f"\n{self.index._formatted_name()}"
        return formatted


#@SequentialIterator
class DataInterface(OperatorIndex):
    """
    A step between the dset.data.DataIndex's and the training pipeline
    """
    def __init__(self, index: OperatorIndex) -> None:
        super().__init__()
        self.index = index

    def get(self, querytime: str | DSETDatetime):
        return self.index[querytime]

    def __getitem__(self, idx):
        return self.get(idx)

    def undo(self, data):
        return data

    def _formatted_name(self, desc: str = None):
        padding = lambda name, length_: name + "".join([" "] * (length_ - len(name)))
        desc = desc or self.__doc__ or "No Docstring"
        desc = desc.replace("\n", "").replace("\t", "").strip()
        formatted = f"{padding(self.__class__.__name__, 30)}{desc}"

        if hasattr(self.index, '_formatted_name'):
            formatted += f"\n{self.index._formatted_name()}"
        return formatted

    @property
    def ignore_sanity(self):
        return False

#@SequentialIterator
class DataIterator(DataStep):
    """
    Provide a way to iterator over data, and catch known errors.

    A child of this class must end the indexes & interfaces section of the data loader.
    """

    def __init__(self, index: DataInterface | OperatorIndex | DataIndex, catch: tuple[Exception] | Exception = None) -> None:
        
        super().__init__(index)
        
        if catch:
            catch = [catch ]if not isinstance(catch, (tuple, list)) else catch
            
            for i, err in enumerate(catch):
                if isinstance(err, str):
                    catch[i] = get_callable(err)
        else:
            catch = []
        self._error_to_catch = tuple(catch)

    def __getattr__(self, key):
        if key == "index":
            raise AttributeError(f"{self.__class__} has no attribute {key}")
        return getattr(self.index, key)

    def set_iterable(
        self,
        start: str | datetime | DSETDatetime,
        end: str | datetime | DSETDatetime,
        interval: int | tuple,
    ):
        """
        Set iteration range for DataInterface

        Parameters
        ----------
        start
            Start date of iteration
        end
            End date of iteration
        interval
            Interval between samples
            Use pandas.to_timedelta notation, (10, 'minute')

        """

        self._interval = time_delta(interval)
        self._start = DSETDatetime(start)
        self._end = DSETDatetime(end)

        self._iterator_ready = True

    def __getitem__(self, idx):
        return self.index[idx]

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
            except self._error_to_catch:
                pass
    
    # def _formatted_name(self):
    #     desc = f"DataIterator for {self.index.__class__.__name__!r}"
    #     formatted = super()._formatted_name(desc)
    #     return formatted
        
class BaseDataOperation(DataStep):
    """
    Provide a way to change the data between a DataInterface and an ML Model.

    Must implement __iter__, __getitem__ & undo
    """

    def __init__(self, index: "DataOperation | DataIterator") -> None:
        self.index = index

    def __getattr__(self, key):
        if key == "index":
            raise AttributeError(f"{self.__class__} has no attribute {key}")
        return getattr(self.index, key)


class DataOperation(DataStep):
    def __init__(self, index: DataStep, apply_func: Callable, undo_func: Callable, *, apply_iterator: bool = True, apply_get: bool = True) -> None:
        """
        Run Operations on Data as it is being used.

        Parameters
        ----------
        index
            Underlying index to use
        """
        super().__init__(index)

        self.apply_func = apply_func
        self.undo_func = undo_func

        self.apply_iterator = apply_iterator
        self.apply_get = apply_get

    def undo(self, data, *args, **kwargs):
        if not self.undo_func is None:
            return self.index.undo(self.undo_func(data))
        return self.index.undo(data, *args, **kwargs)

    def __iter__(self):
        for data in self.index:
            if self.apply_iterator:
                yield self.apply_func(data)
            else:
                yield data

    def __getitem__(self, idx):
        if self.apply_get:
            return self.apply_func(self.index[idx])
        return self.index[idx]


class DataIterationOperator(DataStep):
    """
    A data method to only be applied when iterating.
    """
    def __iter__(self):
        raise NotImplementedError(f"Iteration Operator must be defined in child")

    def __getitem__(self, idx):
        return self.index[idx]
        
    def undo(self, data, *args, **kwargs):
        return self.index.undo(data, *args, **kwargs)

