# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.


from abc import ABCMeta, abstractmethod

from functools import partial
from typing import Callable, Union, Optional, Literal, Type
import warnings


from edit.pipeline_V2.recording import PipelineRecordingMixin
from edit.pipeline_V2.exceptions import PipelineTypeError, PipelineFilterException
from edit.pipeline_V2.warnings import PipelineWarning
from edit.pipeline_V2.parallel import ParallelEnabledMixin


class PipelineStep(PipelineRecordingMixin, ParallelEnabledMixin, metaclass=ABCMeta):
    split_tuples = False
    recognised_types: dict[str, Union[tuple[Type, ...], tuple[Type]]] = {}

    def __init__(
        self,
        split_tuples: Union[dict[str, bool], bool] = False,
        recursively_split_tuples: bool = False,
        recognised_types: Optional[
            Union[
                tuple[Type, ...],
                Type,
                dict[str, Union[tuple[Type, ...], Type]],
            ]
        ] = None,
        response_on_type: Literal["warn", "exception", "ignore", "filter"] = "exception",
    ):
        """
        Base `PipelineStep`
        - all should subclass from this

        Args:
            split_tuples (Union[dict[str, bool], bool], optional):
                Split tuples.
                If dict, allows to distinguish which functions should split tuples.
                Defaults to False.
            recursively_split_tuples when using `_split_tuples_call`. (bool, optional):
                Recursively split tuples when using `_split_tuples_call`. Defaults to False.
            recognised_types (Optional[Union[tuple[Type, ...], Type, dict[str, Union[tuple[Type, ...], Type]]] ], optional):
                Types recognised, can be dictionary to reference different types per function Defaults to None.
            response_on_type (Literal['warn', 'exception', 'ignore', 'filter'], optional):
                Response when invalid type found. Defaults to "exception".
        """
        self.split_tuples = split_tuples
        self.recursively_split_tuples = recursively_split_tuples

        self.recognised_types = recognised_types or {}  # type: ignore
        self.response_on_type = response_on_type

    @abstractmethod
    def run(self, sample):
        raise NotImplementedError()

    def _split_tuples_call(self, sample, *, _function: Union[Callable, str] = "run", **kwargs):
        """
        Split `sample` if it is a tuple and apply `_function` of `self` to each.
        """

        func_name = _function if isinstance(_function, str) else _function.__name__

        to_split = self.split_tuples
        if isinstance(to_split, dict):
            to_split = to_split.get(func_name, False)

        func = partial(
            getattr(self, _function) if isinstance(_function, str) else _function,
            **kwargs,
        )

        if to_split and isinstance(sample, tuple):
            if self.recursively_split_tuples:
                func = partial(self._split_tuples_call, _function=_function, **kwargs)

            return tuple(self.parallel_interface.collect(self.parallel_interface.map(func, sample)))
        return func(sample)

    def check_type(
        self,
        sample,
        *,
        func_name: str,
        override: Optional[tuple[Type, ...]] = None,
    ):
        """
        Check type of `sample` for `func_name`.
        """

        recognised_types = override or self.recognised_types.get(func_name, None)

        if recognised_types is None:  # Check if `func_name` of `self.recognised_types` is et
            return

        if isinstance(sample, recognised_types):
            return

        if self.split_tuples and isinstance(sample, tuple):
            self._split_tuples_call(sample, _function="check_type")

        msg = f"{self.__class__.__name__} received a sample of type: {type(sample)} on {func_name}, when it can only recognise {recognised_types}"
        if self.response_on_type == "exception":
            raise PipelineTypeError(msg)
        elif self.response_on_type == "warn":
            warnings.warn(msg, PipelineWarning)
        elif self.response_on_type == "ignore":
            pass
        elif self.response_on_type == "filter":
            raise PipelineFilterException(sample, msg)
        else:
            raise ValueError(f"Invalid 'response_on_type': {self.response_on_type!r}.")

    def __call__(self, sample):
        return self._split_tuples_call(sample, _function="run")

    def __str__(self):
        return
