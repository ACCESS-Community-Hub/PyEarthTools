# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

import pytest
from edit.pipeline import Operation, Pipeline

from edit.data import Index


class FakeIndex(Index):
    def get(self, idx):
        return idx


class EmptyOperation(Operation):
    def __init__(self):
        super().__init__()

    def apply_func(self, sample):
        return sample

    def undo_func(self, sample):
        return sample


class AllowsOnlyTuples(Operation):
    def __init__(self, split, types):
        super().__init__(split_tuples=split, recognised_types=types)

    def apply_func(self, sample):
        return sample

    def undo_func(self, sample):
        return sample


@pytest.mark.parametrize(
    "split, type",
    [
        (False, tuple),
        (True, int),
    ],
)
def test_recognised_types_success(split, type):
    pipe = Pipeline(FakeIndex(), (EmptyOperation(), EmptyOperation()), AllowsOnlyTuples(split, type))
    assert pipe[1] == (1, 1)


@pytest.mark.parametrize(
    "split, type",
    [
        (True, tuple),
        (False, int),
    ],
)
def test_recognised_types_fails(split, type):
    pipe = Pipeline(FakeIndex(), (EmptyOperation(), EmptyOperation()), AllowsOnlyTuples(split, type))
    with pytest.raises(TypeError):
        assert pipe[1] == (1, 1)
