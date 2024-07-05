# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.
from __future__ import annotations

import edit.utils

from edit.pipeline import Operation
from edit.data import Index


edit.utils.config.set({"pipeline.run_parallel": False})


class FakeIndex(Index):
    """Simply returns the `idx` or `override`."""

    def __init__(self, override: int | None = None):
        self._overrideValue = override
        super().__init__()

    def get(self, idx):
        return self._overrideValue or idx


class MultiplicationOperation(Operation):
    def __init__(self, factor):
        super().__init__(split_tuples=True)
        self.factor = factor

    def apply_func(self, sample):
        return sample * self.factor

    def undo_func(self, sample):
        return sample // self.factor


class MultiplicationOperationUnunifiedable(MultiplicationOperation):
    def undo_func(self, sample):
        return sample + self.factor
