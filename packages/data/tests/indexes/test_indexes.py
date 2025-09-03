from pyearthtools.data import indexes
import pyearthtools.data.archive
from pyearthtools.data.time import Petdt
import pytest
import pathlib


from pyearthtools.data.exceptions import DataNotFoundError


def test_Index(monkeypatch):

    monkeypatch.setattr("pyearthtools.data.indexes.Index.__abstractmethods__", set())

    idx = indexes.Index()

    with pytest.raises(NotImplementedError):
        idx.get()

    with pytest.raises(NotImplementedError):
        idx()


def test_FileSystemIndex(monkeypatch):

    monkeypatch.setattr("pyearthtools.data.indexes.Index.__abstractmethods__", set())
    fsi = indexes.FileSystemIndex()
    with pytest.raises(NotImplementedError):
        fsi.filesystem("anything")

    # Confirm the test begins without a root directory set
    monkeypatch.delattr(pyearthtools.data.archive, "ROOT_DIRECTORIES", raising=False)
    fsi = indexes.FileSystemIndex()
    with pytest.raises(KeyError):
        fsi.ROOT_DIRECTORIES

    # Confirm the FSI responds to the root directory variable correctly
    monkeypatch.setattr(pyearthtools.data.archive, "ROOT_DIRECTORIES", "Hello", raising=False)
    fsi = indexes.FileSystemIndex()
    assert fsi.ROOT_DIRECTORIES == "Hello"

    monkeypatch.setattr(fsi, "filesystem", lambda x: __file__, raising=False)
    assert fsi.search("needle") == __file__

    # Check the current Python file exists
    assert fsi.exists(__file__)
    assert fsi.exists(pathlib.Path(__file__))
    assert fsi.exists({"a": __file__})
    assert fsi.exists((__file__, __file__))

    with pytest.raises(KeyError):
        assert fsi.load(__file__) is not None
        # TODO test actual netcdf file loading

    # FIXME: should this be with pytest.raises(KeyError):
    with pytest.raises(DataNotFoundError):
        assert fsi.get(__file__) is not None
        # TODO test actual netcdf file getting


def test_AdvancedTimeIndex(monkeypatch):

    monkeypatch.setattr("pyearthtools.data.indexes.AdvancedTimeIndex.__abstractmethods__", set())

    data_interval = "day"
    ati = indexes.AdvancedTimeIndex(data_interval)
    dt = Petdt("2023-02")

    monkeypatch.setattr(pyearthtools.data.indexes.Index, "get", lambda x, y: x, raising=False)
    ati.retrieve(dt, use_simple=True)
