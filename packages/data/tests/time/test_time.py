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

from pyearthtools.data.time import Petdt, TimeDelta, TimeRange, TimeResolution
import datetime

from pyearthtools.data import time as pet_time


def test_time_delta():

    # Basic integer
    td = pet_time.time_delta(6)
    assert td.seconds == 6 * 60

    dt_td = datetime.timedelta(minutes=6)
    td = pet_time.time_delta(dt_td)
    assert td.seconds == 6 * 60

    td = pet_time.time_delta("6 minutes")
    assert td.seconds == 6 * 60

    ptd = pet_time.TimeDelta("6 minutes")
    td = pet_time.time_delta(ptd)
    assert td.seconds == 6 * 60

    with pytest.raises(TypeError):
        pet_time.time_delta(56.2)


def test_TimeResolution():

    # Test exception case
    with pytest.raises(TypeError):
        tr = pet_time.TimeResolution(2015)

    # Make a TR from a time specified down to the minute
    tr = pet_time.TimeResolution("2021-02-03T0000")

    # Make a TR from a time specified down to the minute
    tr = pet_time.TimeResolution("2021-02-03T00")
    _as_str = repr(tr)

    # Valid subtration
    tr2 = tr - 1
    assert tr2.resolution == "day"

    # Subtract too much
    with pytest.raises(ValueError):
        tr - 50

    # Add too much
    with pytest.raises(ValueError):
        tr + 50

    with pytest.raises(TypeError):
        tr2 = tr - "dogs"


@pytest.mark.parametrize("timestr, flagstring", [("2021-02-03T0000", "1111100")])
def test_find_components(timestr, flagstring):
    components = pet_time.find_components(timestr)
    flags = [f == "1" for f in flagstring]

    assert list(components.values()) == flags


def test_find_components_exceptions():

    _flagstr = "1111100"
    brokenstr = "zzzz-02-03T0000"
    _longstr = "2021-02-03T0000"

    with pytest.raises(TypeError):

        _components = pet_time.find_components(brokenstr)


@pytest.mark.parametrize(
    "basetime, resolution, expected",
    [
        ("2020-01-01", "day", "2020-01-01"),
        ("2020-01-01", "month", "2020-01"),
        ("2020-01-01", TimeResolution("month"), "2020-01"),
        ("2020-01-01", Petdt("1970-01"), "2020-01"),
        ("2020-01-01", "second", "2020-01-01T00:00:00"),
    ],
)
def test_time_resolution_change(basetime, resolution, expected):
    assert str(Petdt(basetime).at_resolution(resolution)) == expected


def test_time_resolution_compatibility():
    dt1 = Petdt("20230503T071500")
    dt2 = Petdt("20230503T091500")
    assert dt1 != dt2  # Confirm the datetime objects start life different
    assert str(dt1.at_resolution("day")) == "2023-05-03"
    assert str(dt2.at_resolution("day")) == "2023-05-03"
    assert str(dt1.at_resolution("day")) == str(dt2.at_resolution("day"))

    assert dt1.at_resolution("day") == dt2.at_resolution("day")  # But the same at daily resolution


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
    assert str(Petdt(basetime) + TimeDelta(delta)) == expected


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
    assert str(Petdt(time).resolution) == expected_resolution


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
    assert f"{Petdt(time):{str_format}}" == expected
