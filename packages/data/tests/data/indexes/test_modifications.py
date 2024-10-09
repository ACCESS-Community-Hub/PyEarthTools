# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

import pytest

from edit.data.indexes import FakeIndex


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
