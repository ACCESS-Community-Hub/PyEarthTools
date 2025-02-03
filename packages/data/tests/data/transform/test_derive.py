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
import math

from pyearthtools.data.transforms.derive import EquationException, evaluate


@pytest.mark.parametrize(
    "eq, result",
    [
        ("2 + 7", 9),
        ("2 +7", 9),
        ("2+7", 9),
        ("2 - 7", -5),
        ("2 - 7", -5),
        ("2 + -7", -5),
        ("-2 +5", 3),
        ("2 + (5 * 5)", 27),
        ("2 + 5 * 5", 35),
        ("2 + (5 * (7 - 5))", 12),
        ("2 + 5 * 7 - 5", 44),
        ("sqrt(4)", 2),
        ("sqrt 4 * 4", 8),
        ("sqrt(4 * 4)", 4),
        ("cos(pi) + 2 * 5", 5.0),
        ("cos(pi) + (2 * 5)", 9.0),
        ("cos(pi) * (2 * 5)", -10.0),
        ("cos(sin(pi)) * (2 * 5)", 10.0),
    ],
)
def test_evaluate_only_eq(eq, result):
    assert evaluate(eq) == float(result)


@pytest.mark.parametrize(
    "eq, result",
    [
        ("pi", math.pi),
        ("2 * pi", 2 * math.pi),
    ],
)
def test_constants(eq, result):
    assert evaluate(eq) == float(result)
