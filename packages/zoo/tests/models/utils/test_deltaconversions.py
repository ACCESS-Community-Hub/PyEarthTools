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

import pyearthtools.zoo
import pyearthtools.data
import pandas as pd


@pytest.mark.parametrize(
    "value,unit, result",
    [
        (1, "hour", 1),  # Default int unit
        (1, "minutes", 60),  # Default int unit with conversion
        ([1], "hour", 1),
        ([1], "minute", 60),
        ## Str
        ("1-hour", "minutes", 60),  # Conversion
        ("2-hour", "minutes", 120),  # Conversion
        ("2-days", "hour", 48),  # Conversion
        ("2_days", "hour", 48),  # Conversion with other delimiter
        ("2 days", "hour", 48),  # Conversion with other delimiter
        ## Pandas
        (pd.Timedelta(2, "d"), "hour", 48),
        (pd.Timedelta(2, "h"), "hour", 2),
        ## pyearthtools.data
        (pyearthtools.data.TimeDelta(1), "minutes", 1),  # Time Delta defaults to minutes
        (pyearthtools.data.TimeDelta(1, "hour"), "minutes", 60),
        (pyearthtools.data.TimeDelta(1, "day"), "hour", 24),
        (pyearthtools.data.TimeDelta((1, "day")), "hour", 24),
        ## Skipping
        (None, "hour", None),  # Test skipping behaviour
    ],
)
def test_delta_conversion(value, unit, result):
    assert pyearthtools.zoo.utils.delta_conversion(value, unit) == result
