import pyearthtools.data.transforms.normalisation
from pyearthtools.data.transforms.normalisation import default
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


def test_Normaliser():
    n = default.Normaliser("fake index", "start", "end", "month")
    n.check_init_args()
