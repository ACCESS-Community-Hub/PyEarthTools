import xarray as xr
from pyearthtools.data.indexes import cacheIndex


def test_get_size():

    da = xr.DataArray([1, 2, 3, 4, 5])
    assert cacheIndex.get_size(da) != 0
    assert cacheIndex.get_size({"a": [1, 2, 3, 4, 5]})
    assert cacheIndex.get_size([1, 2, 3, 4, 5]) != 0


def test_MemCache():

    mc = cacheIndex.FunctionalMemCacheIndex("pattern", {"a": "b"}, function=str)
    mc.cleanup()
