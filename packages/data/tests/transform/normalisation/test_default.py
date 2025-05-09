import pyearthtools.data.transforms.normalisation
from pyearthtools.data.transforms.normalisation import default
from pyearthtools.data.time import Petdt
import pyearthtools.data.indexes
import xarray as xr
import numpy as np
import pytest

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
def test_open_file(monkeypatch):
    monkeypatch.setattr(pyearthtools.data.transforms.normalisation.default, "open_files", lambda x: sample_da)

    result = default.open_file("pretend_filename.nc")

    assert result is not None


def test_open_non_xarray_file(monkeypatch):
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


# Test Normaliser abstract base class
def test_Normaliser_initialisation(test_Normaliser_default_setup):
    n, ati = test_Normaliser_default_setup

    assert n.retrieval_arguments["start"] == Petdt("2023-02")
    assert n.retrieval_arguments["end"] == Petdt("2023-03")
    assert n.retrieval_arguments["interval"] == "month"


def test_Normaliser_info(test_Normaliser_default_setup):
    n, ati = test_Normaliser_default_setup

    result = n._info_

    assert result is not None
    assert "start" in result
    assert result["start"] == n.retrieval_arguments["start"]


def test_Normaliser_get_average(test_Normaliser_default_setup, monkeypatch):
    n, ati = test_Normaliser_default_setup
    monkeypatch.setattr(ati, "get", lambda x: sample_da)

    result = n.get_average("temperature")

    assert result == 1


def test_Normaliser_get_deviation(test_Normaliser_default_setup, monkeypatch):
    n, ati = test_Normaliser_default_setup
    monkeypatch.setattr(ati, "get", lambda x: sample_da)

    result_mean, result_std = n.get_deviation("temperature")

    assert result_mean == 1
    assert result_std == 0


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


@pytest.mark.parametrize("missing_arg", ["start", "end", "interval"])
def test_Normaliser_missing_retrieval_args(monkeypatch, missing_arg):
    monkeypatch.setattr("pyearthtools.data.indexes.AdvancedTimeIndex.__abstractmethods__", set())

    retrieval_args = {"start": Petdt("2023-02"), "end": Petdt("2023-03"), "interval": "day"}

    ati = pyearthtools.data.indexes.AdvancedTimeIndex("day")

    temp_retrieval_args = retrieval_args.copy()
    temp_retrieval_args.pop(missing_arg)
    with pytest.raises(RuntimeError) as e:
        default.Normaliser(index=ati, **temp_retrieval_args).check_init_args()
    assert missing_arg in str(e.value)


def test_Normaliser_with_override(monkeypatch):
    monkeypatch.setattr("pyearthtools.data.indexes.AdvancedTimeIndex.__abstractmethods__", set())

    ati = pyearthtools.data.indexes.AdvancedTimeIndex("day")
    start = Petdt("2023-02")
    end = Petdt("2023-03")
    interval = "day"

    n = default.Normaliser(ati, start, end, interval, override="True")
    result = n.check_init_args()
    assert result == True


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
