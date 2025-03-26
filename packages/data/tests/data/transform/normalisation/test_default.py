import pyearthtools.data.transforms.normalisation
from pyearthtools.data.transforms.normalisation import default
from pyearthtools.data.time import Petdt
import pyearthtools.data.indexes
import xarray as xr
import numpy as np

sample_da = xr.DataArray(coords={"latitude": [1,2,3,4], "longitude": [1,2,3]},
                         data=np.ones((4,3)))

def test_open_file(monkeypatch):

    monkeypatch.setattr(pyearthtools.data.transforms.normalisation.default, 
                             'open_files', 
                             lambda x: sample_da)

    result = default.open_file("pretend_filename.nc")
    assert result is not None


def test_Normaliser(monkeypatch):

    monkeypatch.setattr("pyearthtools.data.indexes.AdvancedTimeIndex.__abstractmethods__", set())

    data_interval = "day"
    ati = pyearthtools.data.indexes.AdvancedTimeIndex(data_interval)
    monkeypatch.setattr(ati, "get", lambda x: sample_da)
    start = Petdt("2023-02")
    end = Petdt("2023-06")

    n = default.Normaliser(ati, start, end, "month")
    n.check_init_args()

    n.get_average("temperature")
