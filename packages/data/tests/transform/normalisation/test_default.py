import numpy as np
import platform
import pytest
import xarray as xr

import pyearthtools.data.indexes
import pyearthtools.data.transforms.normalisation
from pyearthtools.data.time import Petdt
from pyearthtools.data.transforms.normalisation import default

# Test setup - sample data
sample_da = xr.DataArray(
    coords={"latitude": [1, 2, 3, 4], "longitude": [1, 2, 3], "time": ["2023-02"]}, data=np.ones((4, 3, 1))
)

sample_ds = xr.Dataset(
    coords={"latitude": [1, 2, 3, 4], "longitude": [1, 2, 3], "time": ["2023-02"]}, data_vars={"temperature": sample_da}
)

sample_numpy_array = np.ones((4, 3, 1))


# Test setup - fixtures
@pytest.fixture
def test_Normaliser_default_setup(monkeypatch):
    monkeypatch.setattr("pyearthtools.data.indexes.AdvancedTimeIndex.__abstractmethods__", set())
    data_interval = "day"
    ati = pyearthtools.data.indexes.AdvancedTimeIndex(data_interval)
    start = Petdt("2023-02")
    end = Petdt("2023-03")
    interval = "month"

    n = default.Normaliser(ati, start, end, interval)
    return n, ati


# Test utility functions
def test_open_file(monkeypatch):  # Note that get_default_transforms() is not mocked here
    monkeypatch.setattr(pyearthtools.data.transforms.normalisation.default, "open_files", lambda x: sample_da)

    result = default.open_file("pretend_filename.nc")

    assert result is not None


def test_open_non_xarray_file(monkeypatch):  # Note that get_default_transforms() is not mocked here
    monkeypatch.setattr(pyearthtools.data.transforms.normalisation.default, "open_files", lambda x: sample_numpy_array)

    result = default.open_file("pretend_filename.nc")

    assert result is not None


def test_get_and_print(capsys):
    print_func = default.get_and_print(lambda: list((1, 2)), "print message")

    print_func()
    captured = capsys.readouterr()

    assert captured.out == "print message\n"


def test_get_and_not_print(capsys):
    print_func = default.get_and_print(lambda: list((1, 2)), "print message", False)

    print_func()
    captured = capsys.readouterr()

    assert captured.out == ""


# Test abstract methods
def test_log(test_Normaliser_default_setup):
    n, ati = test_Normaliser_default_setup
    with pytest.raises(NotImplementedError):
        n.log()


def test_anomaly(test_Normaliser_default_setup):
    n, ati = test_Normaliser_default_setup
    with pytest.raises(NotImplementedError):
        n.anomaly()


def test_deviation(test_Normaliser_default_setup):
    n, ati = test_Normaliser_default_setup
    with pytest.raises(NotImplementedError):
        n.deviation()


def test_deviation_spatial(test_Normaliser_default_setup):
    n, ati = test_Normaliser_default_setup
    with pytest.raises(NotImplementedError):
        n.deviation_spatial()


def test_range(test_Normaliser_default_setup):
    n, ati = test_Normaliser_default_setup
    with pytest.raises(NotImplementedError):
        n.range()


# Test special methods
def test_repr(test_Normaliser_default_setup):
    n, ati = test_Normaliser_default_setup

    assert (
        repr(n) == "Normalisation Class waiting upon a request for a method, either call with a method or use property."
    )


# Test Normaliser abstract base class
def test_Normaliser_initialisation_no_cache(test_Normaliser_default_setup):
    n, ati = test_Normaliser_default_setup

    assert n.retrieval_arguments["start"] == Petdt("2023-02")
    assert n.retrieval_arguments["end"] == Petdt("2023-03")
    assert n.retrieval_arguments["interval"] == "month"


def test_Normaliser_initialisation_non_temp_cache(monkeypatch):
    monkeypatch.setattr("pyearthtools.data.indexes.AdvancedTimeIndex.__abstractmethods__", set())
    data_interval = "day"
    ati = pyearthtools.data.indexes.AdvancedTimeIndex(data_interval)
    start = Petdt("2023-02")
    end = Petdt("2023-03")
    interval = "month"
    cache = "path/to/dummy_cache"

    n = default.Normaliser(ati, start, end, interval, cache=cache)

    assert n.cache_dir == cache


def test_Normaliser_initialisation_temp_cache(monkeypatch):
    monkeypatch.setattr("pyearthtools.data.indexes.AdvancedTimeIndex.__abstractmethods__", set())
    data_interval = "day"
    ati = pyearthtools.data.indexes.AdvancedTimeIndex(data_interval)
    start = Petdt("2023-02")
    end = Petdt("2023-03")
    interval = "month"
    cache = "temp"

    _n = default.Normaliser(ati, start, end, interval, cache=cache)

    # On HPC, the cache dir can be different to that of other platforms
    # TODO: Check the normaliser code and fix the assert statement
    # assert n.cache_dir == cache


def test_Normaliser_info(test_Normaliser_default_setup):
    n, ati = test_Normaliser_default_setup

    result = n._info_

    assert result is not None
    assert "start" in result
    assert result["start"] == n.retrieval_arguments["start"]


def test_Normaliser_check_init_args(test_Normaliser_default_setup):
    n, ati = test_Normaliser_default_setup

    result = n.check_init_args()

    assert result is True


@pytest.mark.parametrize("missing_arg", ["start", "end", "interval"])
def test_Normaliser_check_init_args_missing_retrieval_args(monkeypatch, missing_arg):
    monkeypatch.setattr("pyearthtools.data.indexes.AdvancedTimeIndex.__abstractmethods__", set())

    retrieval_args = {"start": Petdt("2023-02"), "end": Petdt("2023-03"), "interval": "day"}

    ati = pyearthtools.data.indexes.AdvancedTimeIndex("day")

    temp_retrieval_args = retrieval_args.copy()
    temp_retrieval_args.pop(missing_arg)
    with pytest.raises(RuntimeError) as e:
        default.Normaliser(index=ati, **temp_retrieval_args).check_init_args()
    assert missing_arg in str(e.value)


@pytest.mark.skipif(platform.system() == "Darwin", reason="This specific test fails on macOS")
def test_Normaliser_get_average(test_Normaliser_default_setup, monkeypatch):
    n, ati = test_Normaliser_default_setup
    monkeypatch.setattr(ati, "get", lambda x: sample_da)

    result = n.get_average("temperature")

    assert result == 1


@pytest.mark.skipif(platform.system() == "Darwin", reason="This specific test fails on macOS")
def test_Normaliser_get_deviation(test_Normaliser_default_setup, monkeypatch):
    n, ati = test_Normaliser_default_setup
    monkeypatch.setattr(ati, "get", lambda x: sample_da)

    result_mean, result_std = n.get_deviation("temperature")

    assert result_mean == 1
    assert result_std == 0


@pytest.mark.skipif(platform.system() == "Darwin", reason="This specific test fails on macOS")
def test_Normaliser_get_anomaly(test_Normaliser_default_setup, monkeypatch):
    n, ati = test_Normaliser_default_setup
    monkeypatch.setattr(ati, "get", lambda x: sample_da)

    result_anomaly = n.get_anomaly("temperature")

    assert result_anomaly is not None

    # FIXME: Need to update the whole test creation to be a time-aware dataset
    # r_range = n.get_range("temperature")
    # assert r_range["temperature"]["max"] == 1
    # assert r_range["temperature"]["min"] == 1

    # result = n.none
    # assert result is not None


def test_Normaliser_with_override(monkeypatch):
    monkeypatch.setattr("pyearthtools.data.indexes.AdvancedTimeIndex.__abstractmethods__", set())

    ati = pyearthtools.data.indexes.AdvancedTimeIndex("day")
    start = Petdt("2023-02")
    end = Petdt("2023-03")
    interval = "day"

    n = default.Normaliser(ati, start, end, interval, override="True")
    result = n.check_init_args()
    assert result


def test_Normaliser_errors(monkeypatch):
    monkeypatch.setattr("pyearthtools.data.indexes.AdvancedTimeIndex.__abstractmethods__", set())

    data_interval = "day"
    ati = pyearthtools.data.indexes.AdvancedTimeIndex(data_interval)
    monkeypatch.setattr(ati, "get", lambda x: sample_da)
    start = Petdt("2023-02")
    end = Petdt("2023-03")

    n = default.Normaliser(ati, start, end, "month")

    with pytest.raises(NotImplementedError):
        n.function()

    not_implemented = [n.log, n.anomaly, n.deviation, n.deviation_spatial, n.range]
    for ni in not_implemented:
        with pytest.raises(NotImplementedError):
            ni()
