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
from abc import abstractmethod
from typing import Any, Literal, Type, Optional, Union


from pyearthtools.pipeline.operation import Operation


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
        Join samples from tuple

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
