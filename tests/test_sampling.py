# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

import pytest
from edit.pipeline import Pipeline, iterators, samplers
from tests.fake_pipeline_steps import FakeIndex


@pytest.mark.parametrize(
    "sampler,length",
    [
        (samplers.Default(), 20),
        (samplers.Random(10), 20),
        ((samplers.Random(10), samplers.Random(10)), 20),
        ((samplers.Random(10), samplers.Default()), 20),
        ((samplers.Default(), samplers.Random(10)), 20),
        ((samplers.Default(), samplers.DropOut(5)), 16),
        ((samplers.Default(), samplers.DropOut(5), samplers.Random(10)), 16),
        ((samplers.RandomDropOut(50), samplers.Random(10)), None),
        ((samplers.RandomDropOut(100), samplers.Random(10)), 0),
        ((samplers.RandomDropOut(0), samplers.Random(10)), 20),
    ],
)
def test_samplers(sampler, length):
    pipe = Pipeline(FakeIndex(), iterator=iterators.Range(0, 20), sampler=sampler)

    if length is not None:
        assert len(list(pipe)) == length, "Length differs from expected"

    iteration_1 = list(pipe)
    iteration_2 = list(pipe)

    assert iteration_1 == iteration_2, "Order is not the same between iterations"
