# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import Any, Union


from pyearthtools.pipeline.operation import Operation
from pyearthtools.pipeline.exceptions import PipelineUnificationException


__all__ = ["Unifier", "Equality"]


class Unifier(Operation, metaclass=ABCMeta):
    """
    Unify samples after a branching point on the undo operation.

    Child class must supply `check_validity`, to determine if the samples can be unified,
     and return an `int` which is used to select a sub_sample to be returned by `undo`.

    If samples are not be unified, `check_validity` should return None.

    Differs from `Spliter` as this is built only to eliminate the tuple created on the `undo`
    with a `BranchingPoint`.
    """

    def __init__(self):
        super().__init__(split_tuples=False, recognised_types={"undo": tuple}, operation="undo")

        self.record_initialisation()

    @abstractmethod
    def check_validity(self, sample: tuple) -> Union[None, int]:  # pragma: no cover
        """
        Check if samples can be unified.

        Raise a `PipelineUnificationException` if not be unified.

        Args:
            sample (tuple):
                Sample's

        Returns:
            (Union[None, int]):
                Which sub_sample to be returned.
                Return `None` if invalid.
        """
        raise NotImplementedError(f"Child class must supply `check_validity` function.")

    def unify(self, sample: tuple) -> Any:
        index = self.check_validity(sample)

        if index is None:
            raise PipelineUnificationException(
                sample,
                f"Elements in tuple cannot be unified with {self.__class__.__name__}",
            )

        return sample[index]

    def apply_func(self, sample):
        return sample

    def undo_func(self, sample: tuple) -> Any:
        return self.unify(sample)


class Equality(Unifier):
    """Check if all elements in tuple are equal"""

    def check_validity(self, sample: tuple) -> Union[None, int]:
        # Check if all elements are equal to the first element
        if all(x == sample[0] for x in sample):
            return 0
