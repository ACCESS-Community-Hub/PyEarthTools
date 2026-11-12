import pytest

import datetime
import random

import site_archive_aws
import pyearthtools.data

from pyearthtools.data import Petdt

VAR_LIST = [
    "temperature_on_pressure_levels",
    "relative_humidity_on_pressure_levels",
    "wind_speed_on_pressure_levels",
    "wind_direction_on_pressure_levels",
]


def _setup_index(var_list):
    ukv_accessor = pyearthtools.data.archive.MOUKV(var_list)
    return ukv_accessor


def _get_random_time():
    current_dt = datetime.datetime.now()
    previous_midnight = datetime.datetime(
        current_dt.year,
        current_dt.month,
        current_dt.day,
        0,
        0,
    )
    # select a date within the coverage of data, which is the past 2 years
    select_dt = previous_midnight - datetime.timedelta(hours=6 * random.randrange(2800))
    return select_dt


def test_ukv_filepaths():
    """
    Test the filpaths functions.
    """
    var_list = VAR_LIST
    ukv_accessor = _setup_index(var_list)

    select_dt = Petdt(_get_random_time())

    file_list = ukv_accessor.filesystem(select_dt)

    assert len(var_list) == len(file_list)
    assert all([v1 in str(f1) for v1, f1 in zip(var_list, file_list)])


@pytest.mark.noci
def test_ukv_load():
    """
    Integration Test of actually retrieving data.
    """
    ukv_index = _setup_index(VAR_LIST)
    select_dt = _get_random_time()
    ukv_ds = ukv_index[select_dt]

    # check the size of the data loaded
    assert ukv_ds["air_temperature"].shape == (1, 33, 970, 1042)


@pytest.mark.xfail(raises=pyearthtools.data.DataNotFoundError)
def test_ukv_novar():
    """
    Test that retrieval fails locally if no variables are specified.
    """
    ukv_index = _setup_index([])
    ukv_ds = ukv_index[_get_random_time()]


@pytest.mark.noci
@pytest.mark.xfail(raises=pyearthtools.data.DataNotFoundError)
def test_ukv_invalid_time():
    """
    Test that retrieval fails if an invalid time is given (data is hourly)
    """
    ukv_index = _setup_index(VAR_LIST)
    ukv_index[_get_random_time() + datetime.timedelta(seconds=600)]


@pytest.mark.noci
@pytest.mark.xfail(raises=pyearthtools.data.DataNotFoundError)
def test_ukv_invalid_var():
    """
    Test that retrieval fails if an invalid variable name is given.
    """
    ukv_index = _setup_index(["nonsense_name"])
    ukv_index[_get_random_time()]
