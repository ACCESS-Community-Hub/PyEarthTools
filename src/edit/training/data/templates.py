import functools
from abc import abstractmethod
from typing import Callable, Union

import yaml
import inspect
from datetime import datetime


from edit.data import DataIndex, OperatorIndex
from edit.data.time import EDITDatetime, time_delta
from edit.training.data.utils import get_indexes, get_callable


class DataStep:
    """
    A step between the edit.data.DataIndex's and the training pipeline
    """

    def __init__(
        self,
        index,
    ):
        self.index = index

    def __getitem__(self, idx):
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()

    def __getattr__(self, key):
        if key == "index":
            raise AttributeError(f"{self.__class__} has no attribute {key}")
        return getattr(self.index, key)

    def __call__(self, idx):
        if isinstance(idx, str | EDITDatetime):
            return self.__getitem__(idx)
        else:
            return self.apply(idx)

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

        if hasattr(self.index, "_formatted_name"):
            formatted += f"\n{self.index._formatted_name()}"
        return formatted

    @property
    def ignore_sanity(self):
        return False


class DataOperation(DataStep):
    def __init__(
        self,
        index,
        apply_func: Callable,
        undo_func: Callable,
        *,
        apply_iterator: bool = True,
        apply_get: bool = True,
    ) -> None:
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


class TrainingOperatorIndex(OperatorIndex):
    """
    EDIT Training Version of edit.data.OperatorIndex
    """

    def __init__(
        self,
        index: "list[TrainingOperatorIndex] | TrainingOperatorIndex",
        *args,
        allow_multiple_index: bool = False,
        **kwargs,
    ) -> None:
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
        if hasattr(self.index, "undo"):
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
        desc = desc.replace("\t", "").strip()
        formatted = f"{padding(self.__class__.__name__, 30)}{desc}"

        if hasattr(self.index, "_formatted_name"):
            formatted += f"\n{self.index._formatted_name()}"
        return formatted


class DataInterface(DataOperation, OperatorIndex):
    """
    A step between the edit.data.DataIndex's and the training pipeline `DataOperation`
    """

    def __init__(self, index: OperatorIndex, **datastep_kwargs) -> None:
        super().__init__(index=index, **datastep_kwargs)

    def get(self, querytime: str | EDITDatetime):
        return self.index[querytime]

    def _formatted_name(self, desc: str = None):
        padding = lambda name, length_: name + "".join([" "] * (length_ - len(name)))
        desc = desc or self.__doc__ or "No Docstring"
        desc = desc.replace("\n", "").replace("\t", "").strip()
        formatted = f"{padding(self.__class__.__name__, 30)}{desc}"

        if hasattr(self.index, "_formatted_name"):
            formatted += f"\n{self.index._formatted_name()}"
        return formatted

    @property
    def ignore_sanity(self):
        return False


# @SequentialIterator
class DataIterator(DataStep):
    """
    Provide a way to iterator over data, and catch known errors.

    A child of this class must end the indexes & interfaces section of the data loader.
    """

    def __init__(
        self,
        index: DataInterface | OperatorIndex | DataIndex,
        catch: tuple[Exception] | Exception = None,
    ) -> None:
        super().__init__(index)

        if catch:
            catch = [catch] if not isinstance(catch, (tuple, list)) else catch

            for i, err in enumerate(catch):
                if isinstance(err, str):
                    catch[i] = get_callable(err)
        else:
            catch = []
        self._error_to_catch = tuple(catch)

    def __getitem__(self, idx: str):
        return self.index[idx]

    def set_iterable(
        self,
        start: str | datetime | EDITDatetime,
        end: str | datetime | EDITDatetime,
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
        self._start = EDITDatetime(start)
        self._end = EDITDatetime(end)

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
            except self._error_to_catch:
                pass


# class BaseDataOperation(DataStep):
#     """
#     Provide a way to change the data between a DataInterface and an ML Model.

#     Must implement __iter__, __getitem__ & undo
#     """

#     def __init__(self, index: "DataOperation | DataIterator") -> None:
#         self.index = index

#     def __getattr__(self, key):
#         if key == "index":
#             raise AttributeError(f"{self.__class__} has no attribute {key}")
#         return getattr(self.index, key)


# class DataOperation(DataStep):
#     def __init__(
#         self,
#         index: DataStep,
#         apply_func: Callable,
#         undo_func: Callable,
#         *,
#         apply_iterator: bool = True,
#         apply_get: bool = True,
#     ) -> None:
#         """
#         Run Operations on Data as it is being used.

#         Parameters
#         ----------
#         index
#             Underlying index to use
#         """
#         super().__init__(index)

#         self.apply_func = apply_func
#         self.undo_func = undo_func

#         self.apply_iterator = apply_iterator
#         self.apply_get = apply_get

#     def undo(self, data, *args, **kwargs):
#         if not self.undo_func is None:
#             return self.index.undo(self.undo_func(data))
#         return self.index.undo(data, *args, **kwargs)

#     def _apply(self, data):
#         return self.apply_func(data)

#     def __iter__(self):
#         for data in self.index:
#             if self.apply_iterator:
#                 yield self.apply_func(data)
#             else:
#                 yield data

#     def __getitem__(self, idx):
#         if self.apply_get:
#             return self.apply_func(self.index[idx])
#         return self.index[idx]


class DataIterationOperator(DataOperation):
    """
    A data method to only be applied when iterating.
    """

    def __init__(self, index) -> None:
        super().__init__(
            index, apply_func=None, undo_func=None, apply_iterator=True, apply_get=False
        )

    def __iter__(self):
        raise NotImplementedError(f"Filter must define Iterator")
