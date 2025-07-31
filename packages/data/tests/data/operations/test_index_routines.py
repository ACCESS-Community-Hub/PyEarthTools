from pyearthtools.data.operations import index_routines

from pyearthtools.data.time import Petdt
from pyearthtools.data.time import TimeDelta

import xarray as xr


def test_mf_series(monkeypatch):

    class DummyData:

        def exists(self, query_time):
            return True

        def search(self, query_time):
            return [str(query_time)]

    def make_dataset(list_of_paths, engine=None, chunks=None, combine_attrs=None, parallel=None):
        return str(list(range(len(list_of_paths))))

    start_time = Petdt("20220101T0000")
    end_time = Petdt("20230101T0000")
    interval = TimeDelta("1 month")
    data_function = DummyData()

    monkeypatch.setattr(xr, "open_mfdataset", make_dataset)

    result = index_routines._mf_series(data_function, start_time, end_time, interval)
    assert result is not None


def test_safe_series(monkeypatch):

    class DummyData:

        def __init__(self):
            self.data_resolution = "year"
            self.data_interval = "month"

        def exists(self, query_time):
            return True

        def search(self, query_time):
            return [str(query_time)]

    def make_dataset(list_of_paths, engine=None, chunks=None, combine_attrs=None, parallel=None):
        return str(list(range(len(list_of_paths))))

    def dummy_series(datafn, start_time, end_time, interval, skip_invalid):

        return xr.Dataset()

    start_time = Petdt("20220101T0000")
    end_time = Petdt("20230101T0000")
    interval = TimeDelta("1 month")
    data_function = DummyData()

    monkeypatch.setattr(xr, "open_mfdataset", make_dataset)
    monkeypatch.setattr(index_routines, "series", dummy_series)

    result = index_routines.safe_series(data_function, start_time, end_time, interval)
    assert result is not None
