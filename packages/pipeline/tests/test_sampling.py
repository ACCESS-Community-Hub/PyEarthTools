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
