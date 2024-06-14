# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations
from abc import abstractmethod
from typing import Any, Literal, Type, TypeVar, Optional, Union


from edit.pipeline_V2.operation import Operation
from edit.pipeline_V2.exceptions import PipelineRuntimeError
from edit.pipeline_V2.decorators import potentialabstractmethod


class Joiner(Operation):
    """
    Join samples after a branching point.

    Child class must implement `join`, and `unjoin`.
    """

    def __init__(
        self,
        *,
        split_tuples: bool = True,
        recursively_split_tuples: bool = False,
        recognised_types: Optional[Union[tuple[Type, ...], Type]] = None,
        response_on_type: Literal["warn", "exception", "ignore"] = "exception",
    ):
        """
        Split samples into tuples

        Args:
            split_tuples (bool, optional):
                Split tuples on `unjoin` operation. Defaults to True.
            recursively_split_tuples (bool, optional):
                Recursively split tuples. Defaults to False.
            recognised_types (Optional[Union[tuple[Type, ...],Type]], optional):
                Types recognised on `unjoin`, `join` automatically has tuples. Defaults to None.
            response_on_type (Literal["warn", "exception", "ignore"], optional):
                Response when invalid type found. Defaults to "exception".
        """
        if recognised_types:
            _recognised_types = {"undo": recognised_types, "apply": tuple}
        else:
            _recognised_types = {"apply": tuple}

        super().__init__(
            split_tuples="undo" if split_tuples else False,
            recursively_split_tuples=recursively_split_tuples,
            operation="both",
            recognised_types=_recognised_types,  # type: ignore
            response_on_type=response_on_type,
        )
        self.record_initialisation()

    @abstractmethod
    def join(self, sample: tuple) -> Any:  # pragma: no cover
        """
        Join method called on `apply`.

        Args:
            sample (tuple):
                Sample to be joined

        Returns:
            (Any):
                Joined `sample`
        """
        return sample

    @abstractmethod
    def unjoin(self, sample: Any) -> tuple:  # pragma: no cover
        """
        Unjoin method called on `undo`.

        If the pipeline is to be fully reversable,
         this should return exactly what was received in `join`.

        If it does not, the pipeline will not be fully reversable.

        Args:
            sample (Any):
                Sample to be split / unjoined.

        Returns:
            (tuple):
                Split / unjoined sample
        """
        return sample

    def apply_func(self, sample):
        return self.join(sample)

    def undo_func(self, sample: tuple):
        return self.unjoin(sample)
