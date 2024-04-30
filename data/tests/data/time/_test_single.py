# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

import pytest

import edit.data

from tests.data.FakeDataIndex import FakeDataIndex, FakeOperatorIndex


@pytest.mark.parametrize(
    "time, select_time, time_size",
    [
        ([0, 10, 20, 30, 40, 50], "2021-01-01T00:00", 1),
        ([0, 10, 20, 30, 40, 50], "2021-01-01T00", 6),
        ([0, 10, 20, 30, 40, 50], "2021-01-01", 6),
    ],
)
def test_index_single(time, select_time, time_size):
    dataindex = FakeDataIndex("test", time, (3, 3), 0, data_interval=(10, "minute"))

    data = dataindex(select_time)
    assert len(data.time) == time_size


@pytest.mark.parametrize(
    "time, select_time, time_size, data_interval",
    [
        ([0, 10, 20, 30, 40, 50], "2021-01-01T00:00", 1, (10, "minutes")),
        ([0, 10, 20, 30, 40, 50], "2021-01-01T00", 6, (10, "minutes")),
        ([0, 10, 20, 30, 40, 50], "2021-01-01", 144, (10, "minutes")),
        ([0], "2021-01-01", 24, (1, "hour")),
    ],
)
def test_operatorindex_single(time, select_time, time_size, data_interval):
    dataindex = FakeOperatorIndex("test", time, (3, 3), 0, data_interval=data_interval)

    data = dataindex(select_time)
    assert len(data.time) == time_size
