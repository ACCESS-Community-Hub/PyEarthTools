# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

import pytest

from edit.pipeline_V2 import Pipeline, iterators, samplers


def replace_getter(arg):
    return arg


def test_iterators(monkeypatch):
    pipe = Pipeline(iterator=iterators.Range(0, 20))
    monkeypatch.setattr(pipe, "_get_initial_sample", replace_getter)
    assert list(pipe) == list(range(0, 20))


@pytest.mark.parametrize(
    "iterator,length",
    [
        (iterators.Range(0, 20), 20),
        (iterators.Predefined([1, 2, 3]), 3),
        (iterators.DateRange("2020-01-01T00", "2020-01-02T00", (1, "hour")), 24),
        (iterators.DateRangeLimit("2020-01-01T00", (1, "hour"), 3), 3),
    ],
)
def test_iterators_many(monkeypatch, iterator, length):
    pipe = Pipeline(iterator=iterator, sampler=samplers.Default())

    monkeypatch.setattr(pipe, "_get_initial_sample", replace_getter)

    if length is not None:
        assert len(list(pipe)) == length, "Length differs from expected"

    iteration_1 = list(pipe)
    iteration_2 = list(pipe)

    assert iteration_1 == iteration_2, "Order is not the same between iterations"
