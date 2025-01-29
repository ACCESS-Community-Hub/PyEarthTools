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


from typing import Optional, Any

import numpy as np

from pyearthtools.pipeline.branching.join import Joiner


class Stack(Joiner):
    """
    Stack a tuple of np.ndarray's

    Currently cannot undo this operation
    """

    _override_interface = ["Delayed", "Serial"]
    _interface_kwargs = {"Delayed": {"name": "Stack"}}

    def __init__(self, axis: Optional[int] = None):
        super().__init__()
        self.record_initialisation()
        self.axis = axis

    def join(self, sample: tuple[Any, ...]) -> np.ndarray:
        """Join sample"""
        return np.stack(sample, self.axis)  # type: ignore

    def unjoin(self, sample: Any) -> tuple:
        return super().unjoin(sample)


class VStack(Joiner):
    """
    Vertically Stack a tuple of np.ndarray's

    Currently cannot undo this operation
    """

    _override_interface = ["Delayed", "Serial"]
    _interface_kwargs = {"Delayed": {"name": "VSplit"}}

    def __init__(self):
        super().__init__()
        self.record_initialisation()

    def join(self, sample: tuple[Any, ...]) -> np.ndarray:
        """Join sample"""
        return np.vstack(
            sample,
        )  # type: ignore

    def unjoin(self, sample: Any) -> tuple:
        return super().unjoin(sample)


class HStack(Joiner):
    """
    Horizontally Stack a tuple of np.ndarray's

    Currently cannot undo this operation
    """

    _override_interface = ["Delayed", "Serial"]
    _interface_kwargs = {"Delayed": {"name": "HSplit"}}

    def __init__(self):
        super().__init__()
        self.record_initialisation()

    def join(self, sample: tuple[Any, ...]) -> np.ndarray:
        """Join sample"""
        return np.hstack(
            sample,
        )  # type: ignore

    def unjoin(self, sample: Any) -> tuple:
        return super().unjoin(sample)


class Concatenate(Joiner):
    """
    Concatenate a tuple of np.ndarray's

    Currently cannot undo this operation
    """

    _override_interface = ["Delayed", "Serial"]
    _interface_kwargs = {"Delayed": {"name": "Concatenate"}}

    def __init__(self, axis: Optional[int] = None):
        super().__init__()
        self.record_initialisation()
        self.axis = axis

    def join(self, sample: tuple[Any, ...]) -> np.ndarray:
        """Join sample"""
        return np.concatenate(sample, self.axis)  # type: ignore

    def unjoin(self, sample: Any) -> tuple:
        return super().unjoin(sample)
