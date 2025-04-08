import xarray as xr

from pyearthtools.data.indexes import cacheIndex
import tempfile


def test_get_size():

    da = xr.DataArray([1, 2, 3, 4, 5])
    assert cacheIndex.get_size(da) != 0
    assert cacheIndex.get_size({"a": [1, 2, 3, 4, 5]})
    assert cacheIndex.get_size([1, 2, 3, 4, 5]) != 0


def test_MemCache():

    mc = cacheIndex.FunctionalMemCacheIndex("PatternIndex", {"transforms": None}, function=str)
    assert mc.size != 0

    assert mc.pattern is not None
    # assert mc.get_hash() is not None

    mc.cleanup()


def test_FileSystemCacheIndex():

    with tempfile.TemporaryDirectory() as tempdir:

        fsci = cacheIndex.FunctionalCacheIndex(tempdir, function=str)
        assert fsci.cache is not None
