# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
# Datetime Overrides

As the default datetime objects contain no reference to the user provided string, the resolution of time
that a user provides is simply lost. [EDITDatetime][edit.data.time.EDITDatetime] introduces this concept as a wrapper around
[pandas.DatetimeIndex][pandas.to_datetime].

Subsequently, a [TimeDelta][edit.data.time.TimeDelta], and an override for range [TimeRange][edit.data.time.TimeRange] are also provided.
"""

from __future__ import annotations

import datetime

import functools
from typing import Any, Generator, Literal, overload, Union

import numpy as np
import pandas as pd
import yaml

from edit.utils import initialisation

VALID_RESOLUTIONS = Literal["year", "month", "day", "hour", "minute", "min", "second", "nanosecond"]
RESOLUTION_COMPONENTS: list[VALID_RESOLUTIONS] = [
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "nanosecond",
]


def multisplit(element: str, splits: tuple[Union[str, int], ...]) -> list[str]:
    """
    Split a str by multiple characters.
    """
    elements = [element]

    for sp in splits:
        full_split = []
        for elem in elements:
            full_split.extend(elem.split(sp))

        elements = full_split
    return elements


def find_components(time: str) -> dict[VALID_RESOLUTIONS, bool]:
    """
    Find Specified Time components in given time str

    Args:
        time (str): String of time, usually in isoformat
                e.g. '2021-02-03T0000

    Returns:
        dict[str, bool]: Dictionary detailing which components were specified

    Examples:
        >>> edit.data.time.find_components('2020-01')
        {'year': True, 'month': True, 'day': False, 'minute': False, 'second': False}
    """

    # Split days and hours
    split_time = time.split("T" if "T" in time else " ")

    seperated_date = split_time[0].split("-")
    if not seperated_date[0].isdigit():
        raise TypeError(
            f"{time!r} does not appear to be an isoformat datetime string. (YEAR-MONTH-DAYTHOUR:MINUTE:SECOND)"
        )

    if seperated_date[0] == split_time[0] and len(split_time[0]) > 4:  # Is specified without '-'
        seperated_date[0] = split_time[0][0:4]  # Save Year

        split_time[0] = split_time[0][4:]  # Get all but year

        while len(split_time[0]) >= 2:  # Continue until exhausted
            seperated_date.append(split_time[0][:2])
            split_time[0] = split_time[0][2:]
        if len(seperated_date) > 3:
            raise ValueError(f"Cannot parse {time!r}, seems to contain more than it should.")

    date_components = [RESOLUTION_COMPONENTS[i] for i in range(len(seperated_date))]

    if len(split_time) > 1:
        seperated_time = []
        if any(map(lambda x: x in split_time[1], (":", "." " "))):
            seperated_time = multisplit(split_time[1], (":", "."))
        else:
            time_size = [2, 2, 2, 9]
            while len(time_size) > 0 and len(split_time[1]) >= time_size[0]:
                seperated_time.append(split_time[1][: time_size[0]])
                split_time[1] = split_time[1][time_size[0] :]
                time_size = time_size[1:]

        time_components = [RESOLUTION_COMPONENTS[i + 3] for i in range(len(seperated_time))]

    else:
        time_components = []

    found_components = [*date_components, *time_components]
    return_values: dict[VALID_RESOLUTIONS, bool] = {comp: comp in found_components for comp in RESOLUTION_COMPONENTS}

    return return_values


def strip_to_common_resolution(component: str) -> str:
    """Remove common suffix for time resolution vernacular"""
    component = component.removesuffix("ly").removesuffix("s")
    return component


@functools.total_ordering
class TimeResolution:
    """Allow comparison of resolution of given times."""

    def __init__(
        self,
        value: dict[VALID_RESOLUTIONS, bool] | VALID_RESOLUTIONS | str | "EDITDatetime" | TimeResolution,
    ):
        """
        Find resolution of EDITDatetime or time string

        Args:
            value (dict[VALID_RESOLUTIONS, bool] | VALID_RESOLUTIONS | str | EDITDatetime | TimeResolution):
                Value to find components from,
                If dict, must be True or False for each component
                If str, must be either date str.

        Raises:
            TypeError: If unable to parse value
        """
        resolution: VALID_RESOLUTIONS | None = None

        if isinstance(value, dict):
            for comp in RESOLUTION_COMPONENTS:
                if value[comp]:
                    resolution = comp

        elif isinstance(value, TimeResolution):
            resolution = value.resolution

        elif isinstance(value, str):
            if strip_to_common_resolution(value) in RESOLUTION_COMPONENTS:
                resolution = strip_to_common_resolution(value)  # type: ignore

            elif EDITDatetime.is_time(value):
                value = EDITDatetime(value)

        if isinstance(value, EDITDatetime):
            resolution = value.resolution.resolution

        if resolution is None:
            raise TypeError(f"Unable to parse {value!r} of type {type(value)}")

        self.resolution: VALID_RESOLUTIONS = resolution

    @property
    def components(self) -> dict[str, bool]:
        comp: dict[str, bool] = {}
        value = True
        for resol in RESOLUTION_COMPONENTS:
            comp[resol] = value
            if resol == self.resolution:
                value = False
        return comp

    def __eq__(self, other: VALID_RESOLUTIONS | TimeResolution) -> bool:
        if isinstance(other, TimeResolution):
            other = other.resolution
        if not isinstance(other, str):
            return NotImplemented
        return RESOLUTION_COMPONENTS.index(self.resolution) == RESOLUTION_COMPONENTS.index(other)

    def __gt__(self, other: VALID_RESOLUTIONS | TimeResolution):
        if isinstance(other, TimeResolution):
            other = other.resolution
        if not isinstance(other, str):
            return NotImplemented
        return RESOLUTION_COMPONENTS.index(self.resolution) > RESOLUTION_COMPONENTS.index(other)

    def __lt__(self, other: VALID_RESOLUTIONS | TimeResolution):
        if isinstance(other, TimeResolution):
            other = other.resolution
        if not isinstance(other, str):
            return NotImplemented
        return RESOLUTION_COMPONENTS.index(self.resolution) < RESOLUTION_COMPONENTS.index(other)

    def __repr__(self):
        return f"TimeResolution({self.resolution!r})"

    def __str__(self) -> VALID_RESOLUTIONS:
        return self.resolution

    def __add__(self, other: int) -> TimeResolution:
        if not isinstance(other, int):
            return NotImplemented

        if other < 0:
            return self - abs(other)

        new_index = RESOLUTION_COMPONENTS.index(self.resolution) + other

        if new_index > len(RESOLUTION_COMPONENTS):
            raise ValueError("Cannot set resolution greater than 'yearly'")

        return TimeResolution(RESOLUTION_COMPONENTS[new_index])

    def __sub__(self, other: int) -> TimeResolution:
        if not isinstance(other, int):
            return NotImplemented

        new_index = RESOLUTION_COMPONENTS.index(self.resolution) - other
        if new_index < 0:
            raise ValueError("Cannot set resolution smaller than 'second'")
        return TimeResolution(RESOLUTION_COMPONENTS[new_index])


class EDITMonthtime:
    def __init__(self, input_month: str | EDITMonthtime | EDITDatetime) -> None:
        raise NotImplementedError()


@functools.total_ordering
class EDITDatetime:
    """Datetime object with awareness of which values were set.

    time must be a str or EDITDatetime for resolution awareness to take effect,
    If str, it must be in isoformat

    Uses pandas.Datetime64 as underlying object.
    But will return str of datetime using only what was set.

    Examples:
        >>> str(edit_datetime('2021-01'))
        "2021-01"
        >>> str(edit_datetime('2021-01-12'))
        "2021-01-12"
    """

    def __init__(self, time: Any, *, resolution: str | TimeResolution | None = None):
        """
        Datetime object with awareness of which values were set.

        Args:
            time (Any):
                Time to get resolution of. Can use 'today' to get today
            resolution (str | TimeResolution | None, optional):
                Override for resolution specification. Defaults to None.
        """
        _pandas_timestep = None
        if isinstance(time, str) and time == "today":
            time = str(datetime.datetime.today().strftime("%Y-%m-%d"))

        elif isinstance(time, EDITDatetime) or isinstance(time, type(self)):
            self._pandas_timestep = time._pandas_timestep
            self.resolution = time.resolution
            return

        elif isinstance(time, (np.datetime64, int)):
            time = str(time)
        elif isinstance(time, pd.Timestamp):
            _pandas_timestep = time

        if isinstance(time, str):
            _pandas_timestep = pd.to_datetime(time)
        if _pandas_timestep is None:
            raise TypeError(f"Cannot parse {type(time)}: {time!r}")
        self._pandas_timestep = _pandas_timestep

        if resolution is None:
            self.resolution = TimeResolution(
                find_components(time) if isinstance(time, str) else {comp: True for comp in RESOLUTION_COMPONENTS}
            )
        else:
            self.resolution = TimeResolution(resolution)

    def datetime64(self, time_unit: str = "ns") -> np.datetime64:
        """
        Get EDITDatetime as a `np.datetime64` in given unit

        Args:
            time_unit (str, optional): Time unit to get datetime64 in. Defaults to "ns".

        Returns:
            np.datetime64: Defined time as a np.datetime64
        """

        return np.datetime64(str(self), time_unit)

    @property
    def qualified(self) -> pd.Timestamp:
        """
        Get fully qualified datetime of EDITDatetime.

        Uses underlying pandas.datetime object

        """
        return self._pandas_timestep

    @property
    def datetime(self) -> datetime.datetime:
        """
        Get `datetime.datetime` object
        """
        return datetime.datetime.fromisoformat(self.qualified.isoformat())

    @staticmethod
    def is_time(time_to_parse: Any) -> bool:
        """
        Check if object can be parsed to a `EDITDatetime`

        Attempts to make `EDITDatetime` but catches all exceptions.

        Args:
            time_to_parse (Any):
                Object to check if can be `EDITDatetime`

        Returns:
            (bool):
                Boolean value of if can be `EDITDatetime`
        """
        try:
            EDITDatetime(time_to_parse)
            return True
        except Exception as e:
            return False

    def at_resolution(
        self,
        resolution: VALID_RESOLUTIONS | EDITDatetime | TimeResolution | TimeDelta | str,
    ) -> EDITDatetime:
        """
        Get EDITDatetime at specified resolution

        Args:
            resolution (VALID_RESOLUTIONS | EDITDatetime | TimeResolution | TimeDelta, optional):
                Temporal Resolution of resulting edit_datetime.

        Raises:
            (KeyError):
                If `resolution` is not recognised

        Returns:
            (EDITDatetime):
                EDITDatetime at given resolution
        """

        if isinstance(resolution, (TimeDelta, EDITDatetime)):
            resolution = resolution.resolution

        elif isinstance(resolution, pd.Timedelta):
            resolution = time_delta_resolution(resolution)

        return EDITDatetime(self.qualified, resolution=resolution)

    def __format__(self, *a):
        if len(a) == 0 or (len(a) == 1 and a[0] == ""):
            return str(self)
        return self._pandas_timestep.__format__(*a)

    def _format(self, type, value):
        if type in ["month", "day", "hour", "minute", "second"]:
            return "%02d" % value
        return str(value)

    def __str__(self):
        # TODO make this better
        if self._pandas_timestep is None:
            return "No Time passed"

        date_part = "-".join(
            [
                self._format(comp, getattr(self._pandas_timestep, comp))
                for comp in list(self.resolution.components.keys())[:3]
                if self.resolution.components[comp]
            ]
        )
        time_part = ":".join(
            [
                self._format(comp, getattr(self._pandas_timestep, comp))
                for comp in list(self.resolution.components.keys())[3:-1]
                if self.resolution.components[comp]
            ]
        )
        if self.resolution.components["nanosecond"]:
            time_part = f"{time_part}.{self._pandas_timestep.nanosecond}"

        return date_part + ("T" if time_part else "") + time_part

    def __repr__(self):
        return f"EDITDatetime({str(self)!r})"

    def __getattr__(self, key):
        if key == "_pandas_timestep":
            raise AttributeError(f"{self.__class__} has no attribute {key!r}")

        if not hasattr(self._pandas_timestep, key):
            raise AttributeError(f"{type(self)} has no attribute {key!r}.")

        attr = getattr(self._pandas_timestep, key)
        return attr
        if callable(attr):

            @functools.wraps(attr)
            def override_func(*args, **kwargs):
                self._pandas_timestep = attr(*args, **kwargs)
                return self

            return override_func
        else:
            return attr

    ###---------------###
    # Math
    ###---------------###
    def __add__(self, other: EDITDatetime | TimeDelta | int) -> EDITDatetime:
        """
        Add to underlying '_pandas_timestep'.

        If timedelta, add and update resolution.
        If int, add to last level of resolution
        """

        resolution = TimeResolution("year")
        if isinstance(other, _MonthTimeDelta):
            return NotImplemented

        if isinstance(other, int):
            if other < 0:
                return self - abs(other)

            new_timestep = self._pandas_timestep
            if self.resolution == "year":
                new_timestep = new_timestep.replace(
                    **{str(self.resolution): getattr(self._pandas_timestep, str(self.resolution)) + other}
                )
            elif self.resolution == "month":
                new_year = getattr(self._pandas_timestep, "year")
                new_month = getattr(self._pandas_timestep, "month") + other

                while new_month > 12:
                    new_month -= 12
                    new_year += 1

                new_timestep = new_timestep.replace(year=new_year, month=new_month)
            else:
                new_timestep += TimeDelta(other, str(self.resolution))
        elif isinstance(other, TimeDelta):
            new_timestep = self._pandas_timestep + other
            resolution: TimeResolution = other.resolution

        else:
            new_timestep = self._pandas_timestep + other

        # if isinstance(new_timestep, pd.Timedelta):
        #     return new_timestep

        new_edit_datetime = EDITDatetime(new_timestep, resolution=max(self.resolution, resolution))
        return new_edit_datetime

    def __radd__(self, other: EDITDatetime | TimeDelta | int) -> EDITDatetime:
        return self.__add__(other)

    @overload
    def __sub__(self, other: TimeDelta | int | datetime.timedelta) -> EDITDatetime: ...

    @overload
    def __sub__(self, other: EDITDatetime) -> TimeDelta: ...

    def __sub__(self, other: EDITDatetime | TimeDelta | int | datetime.timedelta) -> EDITDatetime | TimeDelta:
        """
        Subtract from underlying '_pandas_timestep'.

        If timestep, subtract
        If int, subtract from last level of resolution

        """

        resolution = TimeResolution("year")

        if isinstance(other, _MonthTimeDelta):
            return NotImplemented

        if isinstance(other, int):
            if other < 0:
                return self + abs(other)

            new_timestep = self._pandas_timestep
            if self.resolution == "year":
                new_timestep = new_timestep.replace(
                    **{str(self.resolution): getattr(self._pandas_timestep, str(self.resolution)) - other}
                )
            elif self.resolution == "month":
                new_year = getattr(self._pandas_timestep, "year")
                new_month = getattr(self._pandas_timestep, "month") - other

                while new_month < 1:
                    new_month += 12
                    new_year -= 1

                new_timestep = new_timestep.replace(year=new_year, month=new_month)

            else:
                new_timestep -= time_delta((other, str(self.resolution)))

        elif isinstance(other, TimeDelta):
            new_timestep = self._pandas_timestep - other
            resolution: TimeResolution = other.resolution

        elif isinstance(other, datetime.timedelta):
            new_timestep = self._pandas_timestep - other
            # resolution: TimeResolution = other.resolution

        elif isinstance(other, EDITDatetime):
            new_timestep = TimeDelta(self._pandas_timestep - other._pandas_timestep)
            return new_timestep
        else:
            return NotImplemented

            # new_timestep = self._pandas_timestep - other
            # if isinstance(new_timestep, pd.Timedelta):
            #     return new_timestep

        new_edit_datetime = EDITDatetime(new_timestep, resolution=max(self.resolution, resolution))
        return new_edit_datetime

    def __rsub__(self, other: EDITDatetime) -> EDITDatetime | TimeDelta:
        if not isinstance(other, EDITDatetime):
            return NotImplemented

        new_timestep = other._pandas_timestep - self._pandas_timestep
        if isinstance(new_timestep, pd.Timedelta):
            return TimeDelta(new_timestep)

        resolution = TimeResolution("year")

        new_edit_datetime = EDITDatetime(new_timestep, resolution=max(self.resolution, resolution))
        return new_edit_datetime

    def __hash__(self):
        return hash(str(self))

    ###---------------###
    # Ordering
    ###---------------###
    def __lt__(self, other):
        return self._pandas_timestep < other

    def __gt__(self, other):
        return self._pandas_timestep > other

    def __eq__(self, other):
        return self._pandas_timestep == other


@functools.total_ordering
class TimeDelta:
    def __new__(cls, timedelta: Any = None, *args):
        if args:
            timedelta = (timedelta, *args)

        if isinstance(timedelta, (_MonthTimeDelta, TimeDelta)):
            timedelta = timedelta._input_timedelta  # type: ignore

        month_check = None
        if isinstance(timedelta, (list, tuple)) and len(timedelta) == 2:
            if isinstance(timedelta[1], str):
                month_check = timedelta[1]
        elif isinstance(timedelta, str):
            month_check = multisplit(timedelta, [" ", ",", "-"])[-1]

        if month_check is not None and month_check.removesuffix("s") in [
            "year",
            "month",
        ]:
            cls = _MonthTimeDelta

        return super().__new__(cls)

    def __getnewargs__(self) -> tuple:
        return (self._input_timedelta,)

    def __init__(self, timedelta: Any, *args) -> None:
        """
        Create a TimeDelta Object

        Effectively a wrapper around the `pandas.Timedelta`.

        If no units are supplied, `minutes` is automatically assumed.

        Args:
            timedelta (Any):
                Timedelta arguments, can be int or tuple
            *args (Any):
                Extra Timedelta arguments. If `timedelta` is int, set unit.

        Examples:
            >>> TimeDelta(10, 'days')
            10 days 00:00:00
            >>> TimeDelta((10, 'days'))
            10 days 00:00:00
            >>> TimeDelta(10)
            0 days 00:10:00
        """
        resolution = None
        if args:
            timedelta = (timedelta, *args)
        self._input_timedelta = timedelta

        if isinstance(timedelta, TimeDelta):
            resolution = timedelta._resolution
            self._input_timedelta = timedelta._input_timedelta
            timedelta = timedelta._input_timedelta  # type: ignore

        if isinstance(timedelta, str):
            if timedelta in RESOLUTION_COMPONENTS:
                timedelta = (1, timedelta)
            elif len(multisplit(timedelta, [" ", ",", "-"])) == 2:
                timedelta = tuple(x.strip() for x in multisplit(timedelta, [" ", ",", "-"]))

        if isinstance(timedelta, (list, tuple)) and len(timedelta) == 2:
            if isinstance(timedelta[1], str) and timedelta[1].strip().removesuffix("s") in RESOLUTION_COMPONENTS:
                resolution = TimeResolution(timedelta[1].strip().removesuffix("s"))

            if isinstance(timedelta[0], str):
                timedelta = (float(timedelta[0].strip()), *timedelta[1:])

        elif isinstance(timedelta, int):
            timedelta = (timedelta, "minute")
            resolution = TimeResolution("minute")

        self._resolution = resolution
        self._timedelta: pd.Timedelta = time_delta(timedelta)

    def __hash__(self):
        return hash(str(self))

    @property
    def resolution(self) -> TimeResolution:
        """
        Resolution of the TimeDelta
        """
        if self._resolution:
            return self._resolution
        return time_delta_resolution(self._timedelta)

    @property
    def np_timedelta(self) -> np.timedelta64:
        """
        Numpy timedelta64 of TimeDelta
        """
        return np.timedelta64(self._timedelta)

    @property
    def pd_timedelta(self) -> pd.Timedelta:
        """
        Pandas Timedelta
        """
        return self._timedelta

    @property
    def nanosecond(self) -> float:
        return self._timedelta.total_seconds() * 1e9

    def __str__(self):
        input_delta = self._input_timedelta
        return str(self.pd_timedelta)
        if isinstance(input_delta, str):
            return str(input_delta)
        elif isinstance(input_delta, (tuple, list)):
            return " ".join([f"{x}" for x in input_delta])
        else:
            return str(input_delta)

    def __repr__(self):
        return f"TimeDelta({self._input_timedelta!r})"

    def __getattr__(self, key):
        if key == "_timedelta":
            raise AttributeError(f"{self.__class__} has no attribute {key!r}")
        attr = getattr(self._timedelta, key)
        return attr

    # ------ #
    # Math
    # ------ #
    def __add__(self, other):
        return TimeDelta(self._timedelta + other)

    def __radd__(self, other):
        return other + self._timedelta

    def __sub__(self, other):
        return TimeDelta(self._timedelta - other)

    def __rsub__(self, other):
        return other - self._timedelta

    def __truediv__(self, other):
        result = self._timedelta / other
        if not isinstance(result, pd.Timedelta):
            return result
        return TimeDelta(result)

    def __rtruediv__(self, other):
        result = other / self._timedelta
        if not isinstance(result, pd.Timedelta):
            return result
        return TimeDelta(result)

    def __floordiv__(self, other):
        result = self._timedelta // other
        if not isinstance(result, pd.Timedelta):
            return result
        return TimeDelta(result)

    def __rfloordiv__(self, other):
        result = other // self._timedelta
        if not isinstance(result, pd.Timedelta):
            return result
        return TimeDelta(result)

    def __mul__(self, other):
        return TimeDelta(self._timedelta * other)

    def __rmul__(self, other):
        return TimeDelta(other * self._timedelta)

    # ------ #
    # Comparison
    # ------ #
    def __lt__(self, other):
        return self._timedelta < other

    def __gt__(self, other):
        return self._timedelta > other

    def __eq__(self, other):
        return self._timedelta == other


class _MonthTimeDelta(TimeDelta):
    """Override for month & year based timedeltas"""

    def __init__(self, timedelta: Any, *args):
        if args:
            timedelta = (timedelta, *args)
        if isinstance(timedelta, (_MonthTimeDelta, TimeDelta)):
            timedelta = timedelta._input_timedelta

        if isinstance(timedelta, str):
            if timedelta in RESOLUTION_COMPONENTS:
                timedelta = (1, timedelta)
            elif len(multisplit(timedelta, [" ", ",", "-"])) == 2:
                timedelta = tuple(x.strip() for x in multisplit(timedelta, [" ", ",", "-"]))

        timedelta = (timedelta[0], timedelta[1].removesuffix("s"))
        if timedelta[1] not in ["year", "month"]:
            raise TypeError(
                f"_MonthTimeDelta cannot be calculated for unit {timedelta[1]!r}. Only 'year', 'month' are valid"
            )

        modified_time_delta = list(timedelta)

        _resolution = TimeResolution("month")

        if modified_time_delta[1] == "year":
            _resolution = TimeResolution("year")
            modified_time_delta[0] = int(modified_time_delta[0]) * 12

        super().__init__((int(modified_time_delta[0]) * 30, "days"))
        self._input_timedelta = timedelta

        self.specified_month = int(modified_time_delta[0])
        self._resolution = _resolution

    # def __repr__(self):
    #     return str([self.specified_month, self._input_timedelta[1]])

    def __radd__(self, other):
        if isinstance(other, EDITDatetime):
            new_date = other.at_resolution("month") + self.specified_month
            if isinstance(new_date, EDITDatetime):
                return new_date.at_resolution(max(other.resolution, self.resolution))
            return new_date
        return super().__radd__(other)

    def __rsub__(self, other):
        if isinstance(other, EDITDatetime):
            new_date = other.at_resolution("month") - self.specified_month
            if isinstance(new_date, EDITDatetime):
                return new_date.at_resolution(max(other.resolution, self.resolution))
        return super().__rsub__(other)

    def __mul__(self, other):
        if isinstance(other, int):
            month = int(self.specified_month)
            return TimeDelta(month * other, "month")
        return super().__mul__(other)

    def __rmul__(self, other):
        if isinstance(other, int):
            month = int(self.specified_month)
            return TimeDelta(other * month, "month")
        return super().__rmul__(other)

    def __str__(self):
        input_delta = self._input_timedelta
        if isinstance(input_delta, str):
            return str(input_delta)
        elif isinstance(input_delta, (tuple, list)):
            return " ".join([f"{x}" for x in input_delta])
        else:
            return str(input_delta)


def time_delta(time: Any) -> pd.Timedelta:
    """
    Create a pandas timedelta

    Args:
        time (Any): time of delta, can be:
            int: automatic unit of 'minutes' applied
            tuple: (int, str) with str being unit

    Returns:
        pd.Timedelta: Discovered pandas timedelta
    """

    if isinstance(time, int):
        return pd.to_timedelta(time, "m")
    elif isinstance(time, (list, tuple)):
        try:
            return pd.to_timedelta(*time)
        except TypeError:
            raise TypeError(f"Cannot parse {type(time)}:{time} to pandas.Timedelta.")
    elif isinstance(time, pd.Timedelta):
        return time
    elif isinstance(time, datetime.timedelta):
        return pd.to_timedelta(time)
    elif isinstance(time, str):
        return pd.to_timedelta(time)
    elif isinstance(time, TimeDelta):
        return time._timedelta

    try:
        return pd.to_timedelta(*time)
    except TypeError:
        raise TypeError(f"Cannot parse {type(time)}:{time} to pandas.Timedelta.")


def time_delta_resolution(timedelta: pd.Timedelta | TimeDelta) -> TimeResolution:
    """
    Find resolution of timedelta

    Args:
        timedelta (pd.Timedelta): Given timedelta

    Returns:
        TimeResolution: Resolution of `timedelta`
    """
    last_index = 0
    timedelta = time_delta(timedelta)

    for i in range(len(timedelta.components)):
        if not timedelta.components[i] == 0:
            last_index = i

    resolution = RESOLUTION_COMPONENTS[min(last_index + 2, len(RESOLUTION_COMPONENTS) - 2)]
    return TimeResolution(resolution)


@functools.lru_cache()
def range_samples(start: EDITDatetime, end: EDITDatetime, step: TimeDelta, inclusive: bool = False):
    """Cache generation of time samples"""
    samples = []
    current = start

    if end < start:
        raise ValueError(f"End cannot be smaller then start. {end} < {start}.")

    while current < end:
        samples.append(current)
        current += step
    if inclusive:
        samples.append(end)
    return samples


class TimeRange:
    """
    Get all timesteps between two points at an interval
    """

    def __init__(
        self,
        start: EDITDatetime | str,
        end: EDITDatetime | str,
        step: TimeDelta | int | tuple | str,
        *,
        inclusive: bool = False,
        use_tqdm: bool = False,
        desc: str = "",
        **kwargs: Any,
    ):
        """Generate all timesteps between start & end at step interval.

        Args:
            start (EDITDatetime | str):
                Starting time
            end (EDITDatetime | str):
                Ending Time
            step (TimeDelta | int | tuple):
                Step Interval
            inclusive (bool, optional):
                Include end time. Defaults to False.
            use_tqdm (bool, optional):
                Format iterator with tqdm for interactive use. Defaults to False.
            desc (str, optional):
                Description if `use_tqdm == True`. Defaults to False.
            **kwargs (Any, optional):
                If using tqdm, all kwargs passed through
        """
        if isinstance(end, str) and end == "current":
            end = datetime.datetime.today().strftime("%Y-%m-%d")

        self.start = EDITDatetime(start)
        self.end = EDITDatetime(end)
        self.step = TimeDelta(step)

        self.inclusive = inclusive

        if self.step.resolution > self.start.resolution:
            self.start = self.start.at_resolution(self.step)

        self.use_tqdm = use_tqdm
        self.desc = desc
        self.kwargs = kwargs

    @functools.cache
    def __len__(self):
        return len(range_samples(self.start, self.end, self.step, self.inclusive))

    def __iter__(self) -> Generator[EDITDatetime, None, None]:
        samples = range_samples(self.start, self.end, self.step, self.inclusive)

        if self.use_tqdm:
            from tqdm.auto import tqdm

            with tqdm(total=len(self), **self.kwargs) as pbar:
                for i in samples:
                    pbar.set_description(f"{self.desc}{':' if self.desc else ''} {i}")
                    yield i

                    pbar.update(1)
        else:
            if self.kwargs:
                raise TypeError(f"kwargs given but not using `tqdm`, no execution path for kwargs.")

            for i in samples:
                yield i

    def __repr__(self):
        return f"TimeRange({self.start!r}, {self.end!r}, {self.step!r})"


def TimeDeltaRepresenter(dumper: yaml.Dumper, delta: TimeDelta):

    return dumper.represent_sequence(
        "!TimeDelta",
        delta._input_timedelta,
    )


def TimeDeltaConstructer(loader: yaml.loader.Loader, tag_suffix: str, node):

    return TimeDelta(*loader.construct_sequence(node, deep=True))


initialisation.Dumper.add_multi_representer(TimeDelta, TimeDeltaRepresenter)
initialisation.Loader.add_multi_constructor(f"!TimeDelta", TimeDeltaConstructer)
