import pyearthtools.data.transforms.normalisation
from pyearthtools.data.transforms.normalisation import default
from pyearthtools.data.time import Petdt
import pyearthtools.data.indexes
import xarray as xr
import numpy as np
import pytest

sample_da = xr.DataArray(
    coords={"latitude": [1, 2, 3, 4], "longitude": [1, 2, 3], "time": ["2023-02"]}, data=np.ones((4, 3, 1))
)

sample_ds = xr.Dataset(
    coords={"latitude": [1, 2, 3, 4], "longitude": [1, 2, 3], "time": ["2023-02"]}, data_vars={"temperature": sample_da}
)


def test_open_file(monkeypatch):

    monkeypatch.setattr(pyearthtools.data.transforms.normalisation.default, "open_files", lambda x: sample_da)

    result = default.open_file("pretend_filename.nc")
    assert result is not None


def test_Normaliser(monkeypatch):

    monkeypatch.setattr("pyearthtools.data.indexes.AdvancedTimeIndex.__abstractmethods__", set())

    data_interval = "day"
    ati = pyearthtools.data.indexes.AdvancedTimeIndex(data_interval)
    monkeypatch.setattr(ati, "get", lambda x: sample_da)
    start = Petdt("2023-02")
    end = Petdt("2023-03")

    n = default.Normaliser(ati, start, end, "month")
    n.check_init_args()

    result = n.get_average("temperature")
    assert result == 1

    r_mean, r_std = n.get_deviation("temperature")
    assert r_mean == 1
    assert r_std == 0

    r_anomaly = n.get_anomaly("temperature")
    assert r_anomaly is not None

    # FIXME: Need to update the whole test creation to be a time-aware dataset
    # r_range = n.get_range("temperature")
    # assert r_range["temperature"]["max"] == 1
    # assert r_range["temperature"]["min"] == 1

    result = n.none
    assert result is not None


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
