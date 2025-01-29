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


import pytest

from pyearthtools.pipeline import Pipeline, iterators, samplers
from tests.fake_pipeline_steps import FakeIndex


def test_iterators():
    pipe = Pipeline(FakeIndex(), iterator=iterators.Range(0, 20))
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
def test_iterators_many(iterator, length):
    pipe = Pipeline(FakeIndex(), iterator=iterator, sampler=samplers.Default())

    if length is not None:
        assert len(list(pipe)) == length, "Length differs from expected"

    iteration_1 = list(pipe)
    iteration_2 = list(pipe)

    assert iteration_1 == iteration_2, "Order is not the same between iterations"
