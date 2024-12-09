# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

import pytest

from pyearthtools.data.time import pyearthtoolsDatetime, TimeDelta, TimeRange, TimeResolution


@pytest.mark.parametrize(
    "basetime, resolution, expected",
    [
        ("2020-01-01", "day", "2020-01-01"),
        ("2020-01-01", "month", "2020-01"),
        ("2020-01-01", TimeResolution("month"), "2020-01"),
        ("2020-01-01", pyearthtoolsDatetime("1970-01"), "2020-01"),
        ("2020-01-01", "second", "2020-01-01T00:00:00"),
    ],
)
def test_time_resolution_change(basetime, resolution, expected):
    assert str(pyearthtoolsDatetime(basetime).at_resolution(resolution)) == expected


@pytest.mark.parametrize(
    "basetime, delta, expected",
    [
        ("2020-01-01", (1, "days"), "2020-01-02"),
        ("2020-01-04", (1, "month"), "2020-02-04"),
        ("2020-01", (1, "month"), "2020-02"),
        ("2020-01-23", (12, "month"), "2021-01-23"),
        ("2020-01", (12, "month"), "2021-01"),
        ("2020", (12, "month"), "2021-01"),
        ("2020", (1, "year"), "2021"),
        ("2020", (100, "year"), "2120"),
    ],
)
def test_time_addition(basetime, delta, expected):
    assert str(pyearthtoolsDatetime(basetime) + TimeDelta(delta)) == expected


@pytest.mark.parametrize(
    "start, end, interval, length",
    [
        ("2020-01-01", "2020-01-01", (1, "days"), 0),
        ("2020-01-01", "2020-01-02", (1, "days"), 1),
        ("2020-01-01T00:00", "2020-01-01T01:00", (10, "minute"), 6),
        ("2020-01-01T00:00", "2020-01-02T00", (10, "minute"), 144),
        ("2020-01-01T00:00", "2020-01-02T00", (60, "minute"), 24),
        ("2020-01-01T00:00", "2020-01-02T00", (1, "hour"), 24),
        ("2020-01", "2021-01", (1, "month"), 12),
        ("2020", "2023", (1, "year"), 3),
    ],
)
def test_range(start, end, interval, length):
    assert len([time for time in TimeRange(start, end, interval)]) == length


@pytest.mark.parametrize(
    "time, expected_resolution",
    [
        ("2020-01-01", "day"),
        ("20200101", "day"),
        ("2020-01-01T", "day"),
        ("20200101T", "day"),
        ("2020-01", "month"),
        # ("202001", "month"),
        ("2020", "year"),
        ("2020-01-01T00", "hour"),
        ("20200101T00", "hour"),
        ("2020-01-01T0000", "minute"),
        ("2020-01-01T00:00", "minute"),
        ("20200101T0000", "minute"),
        ("2020-01-01T000000", "second"),
        ("2020-01-01T00:00:00", "second"),
        ("20200101T000000", "second"),
    ],
)
def test_resolution(time, expected_resolution):
    assert str(pyearthtoolsDatetime(time).resolution) == expected_resolution


@pytest.mark.parametrize(
    "init_resolution, addition, expected_resolution",
    [
        ("year", 1, "month"),
        ("month", 1, "day"),
        ("year", 2, "day"),
        ("month", -1, "year"),
        ("day", -2, "year"),
        ("hour", -1, "day"),
        ("hour", 1, "minute"),
    ],
)
def test_added_resolution(init_resolution, addition, expected_resolution):
    assert TimeResolution(init_resolution) + addition == expected_resolution


@pytest.mark.parametrize(
    "time, str_format, expected",
    [
        ("2020-01-01", "", "2020-01-01"),
        ("2020-01-01", "%Y", "2020"),
    ],
)
def test_f_str(time, str_format, expected):
    assert f"{pyearthtoolsDatetime(time):{str_format}}" == expected
