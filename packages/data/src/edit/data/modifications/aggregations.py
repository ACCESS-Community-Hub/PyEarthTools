# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
Aggregation based `Modification's`
"""

from __future__ import annotations

from typing import Literal
import xarray as xr

from edit.data.time import EDITDatetime, TimeDelta, TimeResolution, TimeRange
from edit.data.indexes.utilities.dimensions import identify_time_dimension

from edit.data.modifications import Modification, register_modification


class Aggregation(Modification):
    """
    Root class for the creation of an aggregated variable

    Cannot be directly used.

    time dimension will be renamed `aggregate_dim` and is expected to be aggregated over for `single`.
    """

    def __init__(self, period: str, align: Literal["past"] = "past", **kwargs):
        """
        Setup aggregator

        Args:
            period (str):
                Period to aggregate over
                Used here to extend time
            inclusive (bool, optional):
                Include end time.
                Defaults to False.
        """
        super().__init__(**kwargs)
        self._period = period
        self._align = align

    def _parse_period(self) -> TimeDelta:
        """
        Get `period` parsed into a `TimeDelta`.

        Allows 'steps' to be used to directly reference the data's resolution.
        """
        if "steps" in self._period:
            period = self._period.removesuffix("steps").strip()
            return TimeDelta(
                (self._data_index.data_interval or TimeDelta(1, str(self._data_index.data_resolution or "hour")))
                * int(period)
            )

        val, period = self._period.split(" ")
        return TimeDelta(int(val), str(TimeResolution(period)))

    def _single(self, time) -> xr.Dataset:
        """Get data needed for an aggregation around a single time

        Does not perform the aggreagtion tho
        """
        time = EDITDatetime(time)
        period = self._parse_period()

        if self._align == "past":
            start = time - period
            end = time

        series = self._data_index.series(
            start,
            end,
        )
        time_dim = identify_time_dimension(series)
        series = series.rename({time_dim: "aggregate_dim"}).expand_dims({time_dim: [end.datetime64()]})
        series[time_dim].attrs = series.aggregate_dim.attrs
        series[time_dim].encoding = series.aggregate_dim.encoding
        return series

    def _series(self, start, end, interval) -> xr.Dataset:
        """Get data needed for an aggregation around a series

        Does not perform the aggreagtion tho
        """
        start = EDITDatetime(start)
        end = EDITDatetime(end)

        period = self._parse_period()
        start_adjusted = start

        if self._align == "past":
            start_adjusted = start - period

        series = self._data_index.series(start_adjusted, end, inclusive=True)

        # time_values = list(map(lambda x: x.datetime64(), TimeRange(start_adjusted, end, self._data_index.data_interval))) # type: ignore

        # time_dim = identify_time_dimension(series)
        # series = series.rename({time_dim: 'aggregate_dim'}).expand_dims({time_dim: time_values})
        return series


@register_modification("aggregate")
class AggregationGeneral(Aggregation):
    """
    Create a general aggregation over time variable.

    Aggregates as a rolling window of size `period`

    Usage:
    -  !aggregation[method: 'max', period: "6 hours"]
    """

    def __init__(self, method: str, period: str, align: Literal["past"] = "past", **kwargs):
        """
        General aggregation

        Args:
            method (str):
                Method name to use
            period (str):
                Period to apply `method` over
            inclusive (bool, optional):
                Include end time.
                Defaults to False.
        """
        super().__init__(period, align, **kwargs)
        self._method = method

    @property
    def attribute_update(self):
        """Attributes to update on variable"""
        return {"Aggregation": f"{self._method} over {self._parse_period()}"}

    def __repr__(self):
        return f"{self._method} of {self._variable!r} over {self._parse_period()}"

    def single(self, time) -> xr.Dataset:
        data_series = self._single(time)
        return getattr(data_series, self._method)(dim="aggregate_dim")

    def series(self, start, end, interval) -> xr.Dataset:
        data_series = self._series(start, end, interval)
        period = self._parse_period()
        time_dim = identify_time_dimension(data_series)

        return getattr(data_series.rolling({time_dim: int(period / interval)}), self._method)().sel(
            {time_dim: slice(str(start), str(end))}
        )


@register_modification("mean")
class Mean(Aggregation):
    """
    Create a mean over time variable

    Averages as a rolling window of size `period`

    Usage:
    -  !mean[period: "6 hours"]
    """

    @property
    def attribute_update(self):
        """Attributes to update on variable"""
        return {"average": f"Averaged over {self._parse_period()}"}

    def __repr__(self):
        return f"Average {self._variable!r} over {self._parse_period()}"

    def single(self, time) -> xr.Dataset:
        data_series = self._single(time)
        return data_series.mean(dim="aggregate_dim")

    def series(self, start, end, interval) -> xr.Dataset:
        data_series = self._series(start, end, interval)
        period = self._parse_period()
        time_dim = identify_time_dimension(data_series)

        return (
            data_series.rolling({time_dim: int(period / interval)}).mean().sel({time_dim: slice(str(start), str(end))})
        )


@register_modification("accumulate")
class Accumulate(Aggregation):
    """
    Create an accumlated over time variable

    Accumulates as a rolling window of size `period`

    Usage:
    -  !accumulate[period: "6 hours"]
    """

    @property
    def attribute_update(self):
        """Attributes to update on variable"""
        return {"accumulation": f"Accumulated over {self._parse_period()}"}

    def __repr__(self):
        return f"Accumulate {self._variable!r} over {self._parse_period()}"

    def single(self, time) -> xr.Dataset:
        data_series = self._single(time)
        return data_series.sum(dim="aggregate_dim")

    def series(self, start, end, interval) -> xr.Dataset:
        data_series = self._series(start, end, interval)
        time_dim = identify_time_dimension(data_series)

        period = self._parse_period()
        return (
            data_series.rolling({time_dim: int(period / interval)}).sum().sel({time_dim: slice(str(start), str(end))})
        )
