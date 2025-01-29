# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
