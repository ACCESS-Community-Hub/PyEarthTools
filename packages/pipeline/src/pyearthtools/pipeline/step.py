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


from pyearthtools.pipeline.recording import PipelineRecordingMixin
from pyearthtools.pipeline.exceptions import PipelineTypeError, PipelineFilterException
from pyearthtools.pipeline.warnings import PipelineWarning
from pyearthtools.pipeline.parallel import ParallelEnabledMixin


class PipelineStep(PipelineRecordingMixin, ParallelEnabledMixin, metaclass=ABCMeta):
    """
    Core step of a pipeline

    Properties:
        _import: Optional str specifying import location of the subclassed step.
        _override_interface: Override for interface ordering.
    """

    split_tuples: Union[dict[str, bool], bool] = False
    recognised_types: dict[str, Union[tuple[Type, ...], tuple[Type]]]

    _import: Optional[str] = None  # Module to import to find this class
    _override_interface = ["Delayed", "Serial"]  # Order of interfaces to try.

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
        - all steps should subclass from this

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

    def _split_tuples_call(
        self,
        sample,
        *,
        _function: Union[Callable, str] = "run",
        override_for_split: Optional[bool] = None,
        allow_parallel: bool = True,
        **kwargs,
    ):
        """
        Split `sample` if it is a tuple and apply `_function` of `self` to each.
        """

        if allow_parallel:
            parallel_interface = self.parallel_interface
        else:
            parallel_interface = self.get_parallel_interface("Serial")

        func_name = _function if isinstance(_function, str) else _function.__name__

        to_split = override_for_split or self.split_tuples
        if isinstance(to_split, dict):
            to_split = to_split.get(func_name, False)

        func = partial(
            getattr(self, _function) if isinstance(_function, str) else _function,
            **kwargs,
        )

        if to_split and isinstance(sample, tuple):
            func = partial(
                self._split_tuples_call, _function=_function, override_for_split=self.recursively_split_tuples, **kwargs
            )
            return tuple(parallel_interface.collect(parallel_interface.map(func, sample)))

        return parallel_interface.collect(parallel_interface.submit(func, sample))

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

        try:
            from dask.delayed import Delayed

            recognised_types = (*recognised_types, Delayed)
        except (ImportError, ModuleNotFoundError):
            pass

        if isinstance(sample, recognised_types):
            return

        if self.split_tuples and isinstance(sample, tuple):
            self._split_tuples_call(sample, _function="check_type", func_name=func_name)

        msg = f"'{self.__class__.__module__}.{self.__class__.__qualname__}' received a sample of type: {type(sample)} on {func_name}, when it can only recognise {recognised_types}"

        if self.response_on_type == "exception":
            raise PipelineTypeError(msg)
        elif self.response_on_type == "warn":
            warnings.warn(msg, PipelineWarning)
            return
        elif self.response_on_type == "ignore":
            return
        elif self.response_on_type == "filter":
            raise PipelineFilterException(sample, msg)
        raise ValueError(f"Invalid 'response_on_type': {self.response_on_type!r}.")

    def __call__(self, sample):
        self.check_type(sample, func_name="run")
        return self._split_tuples_call(sample, _function="run", allow_parallel=False)
