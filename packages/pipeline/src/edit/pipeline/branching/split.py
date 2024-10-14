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


from edit.pipeline.operation import Operation

T = TypeVar("T", Any, Any)


class Spliter(Operation):
    """
    Split samples.

    Useful prior to `PipelineBranchingPoint` set up to `map`.
    Can therefore assign subsets of the sample to various branches.

    Child class must implement `split`, and `join`.
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
                Split tuples on `split` operation. Defaults to True.
            recursively_split_tuples (bool, optional):
                Recursively split tuples. Defaults to False.
            recognised_types (Optional[Union[tuple[Type, ...],Type]], optional):
                Types recognised on `split`, `join` automatically has tuples. Defaults to None.
            response_on_type (Literal["warn", "exception", "ignore"], optional):
                Response when invalid type found. Defaults to "exception".
        """
        if recognised_types:
            _recognised_types = {"apply": recognised_types, "undo": tuple}
        else:
            _recognised_types = {"undo": tuple}

        super().__init__(
            split_tuples="apply" if split_tuples else False,
            recursively_split_tuples=recursively_split_tuples,
            operation="both",
            recognised_types=_recognised_types,  # type: ignore
            response_on_type=response_on_type,
        )
        self.record_initialisation()

    @abstractmethod
    def split(self, sample: T) -> tuple[T, ...]:  # pragma: no cover
        """
        Split method called on `apply`.

        Args:
            sample (Any):
                Sample to be split into tuple

        Returns:
            (tuple[Any, ...]):
                Split `sample`
        """
        return sample

    @abstractmethod
    def join(self, sample: tuple[T, ...]) -> T:  # pragma: no cover
        """
        Join method called on `undo`.

        If the pipeline is to be fully reversable,
         this should return exactly what was received in `split`.

        If it does not, the pipeline will not be fully reversable.

        Args:
            sample (tuple[Any, ...]):
                Sample to be joined.

        Returns:
            (Any):
                Joined `sample`
        """
        return sample  # type: ignore

    def apply_func(self, sample):
        return self.split(sample)

    def undo_func(self, sample: tuple):
        return self.join(sample)
