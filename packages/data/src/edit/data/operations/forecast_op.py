# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations


import xarray as xr


from edit.data.exceptions import DataNotFoundError, InvalidIndexError, InvalidDataError
from edit.data.time import EDITDatetime, TimeDelta, TimeRange
from edit.data.transforms.transform import Transform, TransformCollection
from edit.data.warnings import IndexWarning

from edit.data.operations import index_routines
from edit.data.operations.utils import identify_time_dimension


def forecast_series(
    DataFunction: "Index",
    start: str | EDITDatetime,
    end: str | EDITDatetime,
    interval: tuple[float, str] | TimeDelta,
    *,
    lead_time: tuple[float, str] | TimeDelta = None,
    inclusive: bool = False,
    skip_invalid: bool = False,
    transforms: Transform | TransformCollection = TransformCollection(),
    verbose: bool = False,
) -> xr.Dataset:
    if lead_time is not None:
        return forecast_select_time(
            DataFunction,
            start,
            end,
            interval,
            lead_time=lead_time,
            inclusive=inclusive,
            skip_invalid=skip_invalid,
            transforms=transforms,
            verbose=verbose,
        )

    return forecast_as_basetime(
        DataFunction,
        start,
        end,
        interval,
        inclusive=inclusive,
        skip_invalid=skip_invalid,
        transforms=transforms,
        verbose=verbose,
    )


def forecast_as_basetime(
    DataFunction: "Index",
    start: str | EDITDatetime,
    end: str | EDITDatetime,
    interval: tuple[float, str] | TimeDelta,
    *,
    inclusive: bool = False,
    skip_invalid: bool = False,
    transforms: Transform | TransformCollection = TransformCollection(),
    verbose: bool = False,
):
    """
    Forecast series concating by basetime
    """

    def preprocess(ds: xr.Dataset):
        time_dim = identify_time_dimension(ds)
        time = ds[time_dim].data[0]

        ds = ds.assign_coords(basetime=[time])
        ds[time_dim] = [t - time for t in ds[time_dim].values]
        ds = ds.rename({time_dim: "leadtime"})
        return ds

    return index_routines.series(
        DataFunction,
        start,
        end,
        interval,
        inclusive=inclusive,
        transforms=transforms,
        verbose=verbose,
        subset_time=False,
        preprocess=preprocess,
        skip_invalid=skip_invalid,
    )


def forecast_select_time(
    DataFunction: "Index",
    start: str | EDITDatetime,
    end: str | EDITDatetime,
    interval: tuple[float, str] | TimeDelta,
    lead_time: tuple[float, str] | TimeDelta,
    *,
    inclusive: bool = False,
    skip_invalid: bool = False,
    transforms: Transform | TransformCollection = TransformCollection(),
    verbose: bool = False,
):
    """
    Forecast Series operation selecting a particular lead time
    """
    start = EDITDatetime(start)
    end = EDITDatetime(end)
    interval = TimeDelta(interval)

    lead_time = TimeDelta(lead_time)

    if inclusive:
        end += interval

    data = []

    for time in TimeRange(start, end, interval, use_tqdm=verbose):
        try:
            loaded_data = DataFunction(time, transforms=transforms)
            retrieval_time = (time + lead_time).at_resolution(lead_time).datetime64()

            if retrieval_time not in loaded_data.time:
                raise DataNotFoundError(f"{retrieval_time} not in time. {loaded_data.time}")

            loaded_data = loaded_data.sel(time=retrieval_time)
            data.append(loaded_data)

        except (DataNotFoundError, InvalidDataError) as e:
            if not skip_invalid:
                raise e
            else:
                pass

    data = xr.concat(data, dim="time")
    data.attrs["leadtime"] = str(lead_time)
    return data
