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
    moglobal_index = pyearthtools.data.archive.MOGlobal10km(var_list)
    return moglobal_index


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


def test_moglobal_filepaths():
    """
    Test the filpaths functions.
    """
    var_list = VAR_LIST
    ukv_accessor = _setup_index(var_list)

    select_dt = Petdt(_get_random_time())

    file_list = ukv_accessor.filesystem(select_dt)

    assert len(var_list) == len(file_list)
    assert all([v1 in str(f1) for v1, f1 in zip(var_list, file_list)])


# @pytest.mark.slow
@pytest.mark.noci
def test_moglobal_load():
    """
    Integration Test of actually retrieving data.
    """
    moglobal_index = _setup_index(VAR_LIST)
    select_dt = _get_random_time()
    moglobal_ds = moglobal_index[select_dt]

    # check the size of the data loaded
    assert moglobal_ds["air_temperature"].shape == (1, 37, 1920, 2560)


@pytest.mark.xfail(raises=pyearthtools.data.DataNotFoundError)
def test_moglobal_novar():
    """
    Test that retrieval fails locally if no variables are specified.
    """
    moglobal_index = _setup_index([])
    moglobal_ds = moglobal_index[_get_random_time()]


@pytest.mark.noci
@pytest.mark.xfail(raises=pyearthtools.data.DataNotFoundError)
def test_moglobal_invalid_time():
    """
    Test that retrieval fails if an invalid time is given (data is hourly)
    """
    moglobal_index = _setup_index(VAR_LIST)
    moglobal_index[_get_random_time() + datetime.timedelta(seconds=600)]


@pytest.mark.noci
@pytest.mark.xfail(raises=pyearthtools.data.DataNotFoundError)
def test_moglobal_invalid_var():
    """
    Test that retrieval fails if an invalid variable name is given.
    """
    moglobal_index = _setup_index(["nonsense_name"])
    moglobal_index[_get_random_time()]
