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

from pyearthtools.data.indexes import FakeIndex


@pytest.mark.parametrize(
    "period, value",
    [
        ("6 steps", 6),
        ("6 hours", 6),
        ("1 day", 24),
    ],
)
def test_accumulate(period, value):
    index = FakeIndex(
        variable=f'!accumulate[period:"{period}"]:data',
        interval=(1, "hour"),
        random=False,
        max_value=1,
        data_size=(2, 2),
    )
    assert index["2020-01-01T00"]["data"].mean().values == value


def test_rename():
    index = FakeIndex(
        variable='!accumulate[period:"2 hours"]:data>accum_data',
        interval=(1, "hour"),
        random=False,
        max_value=1,
        data_size=(2, 2),
    )
    assert "accum_data" in index["2020-01-01T00"]


@pytest.mark.parametrize(
    "period",
    [
        "6 steps",
        "6 hours",
    ],
)
def test_accumulate_manual(period):
    index = FakeIndex(
        variable=f'!accumulate[period:"{period}"]:data',
        interval=(1, "hour"),
        random=False,
        max_value=1,
        data_size=(2, 2),
    )
    index_manual = FakeIndex(
        variable=f"data",
        interval=(1, "hour"),
        random=False,
        max_value=1,
        data_size=(2, 2),
    )
    assert (
        index["2020-01-01T00"]["data"].mean().values
        == index_manual.series("2020-01-01T00", "2020-01-01T06").sum(dim="time")["data"].mean().values
    )


@pytest.mark.parametrize(
    "period, value",
    [
        ("6 steps", 1),
        ("6 hours", 1),
        ("1 day", 1),
    ],
)
def test_average(period, value):
    index = FakeIndex(
        variable=f'!mean[period:"{period}"]:data',
        interval=(1, "hour"),
        random=False,
        max_value=1,
        data_size=(2, 2),
    )
    assert index["2020-01-01T00"]["data"].mean().values == value
