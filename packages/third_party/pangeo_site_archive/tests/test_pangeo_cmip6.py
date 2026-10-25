import pathlib

import pandas
import pyearthtools.data
import pytest
import site_archive_pangeo
import xarray
from pyearthtools.data import Petdt
from site_archive_pangeo import PangeoCMIP6

CATALOG = str(pathlib.Path(__file__).parent / "data" / "mini_catalog.csv")
TIME = Petdt("2000-02")
TAS_URL = "gs://test-bucket/tas/gn/v20200101/"
PR_URL = "gs://test-bucket/pr/gn/v20190101/"


def _index(variables, **kwargs):
    kwargs = {"source_id": "TEST-MODEL", "experiment_id": "historical", **kwargs}
    return PangeoCMIP6(variables, catalog=CATALOG, **kwargs)


def _fake_open_zarr(seen):
    """Stand-in for xarray.open_zarr, returning a tiny dataset for each store URL."""
    times = pandas.date_range("2000-01-01", periods=3, freq="MS")

    def fake(url, **kwargs):
        seen[url] = kwargs
        name = url.rstrip("/").split("/")[-3]
        return xarray.Dataset(
            {name: ("time", [1.0, 2.0, 3.0]), "time_bnds": ("time", [0.0, 0.0, 0.0])},
            coords={"time": times},
        )

    return fake


def test_registered_in_archive():
    assert pyearthtools.data.archive.PangeoCMIP6 is PangeoCMIP6
    assert pyearthtools.data.archive.pangeo is site_archive_pangeo


def test_sample_construction_needs_no_network(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("the catalog must not be read at construction")

    monkeypatch.setattr(pandas, "read_csv", no_network)
    assert isinstance(PangeoCMIP6.sample(), PangeoCMIP6)


def test_description():
    assert "Documentation" in _index(["tas"])._desc_


def test_filesystem_picks_latest_version():
    assert _index(["tas", "pr"]).filesystem(TIME) == {"tas": TAS_URL, "pr": PR_URL}


def test_single_variable_as_string():
    assert _index("pr").filesystem(TIME) == {"pr": PR_URL}


def test_experiment_is_filtered():
    stores = _index(["tas"], experiment_id="ssp585").filesystem(TIME)
    assert stores == {"tas": "gs://test-bucket/ssp/tas/gn/v20190101/"}


def test_catalog_is_read_once(monkeypatch):
    calls = []
    real_read_csv = pandas.read_csv

    def counting_read_csv(*args, **kwargs):
        calls.append(args)
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr(pandas, "read_csv", counting_read_csv)
    index = _index(["tas"])
    index.filesystem(TIME)
    index.filesystem(TIME)
    assert len(calls) == 1


def test_no_variables_rejected():
    with pytest.raises(ValueError, match="variable"):
        _index([])


def test_unknown_variable_raises():
    with pytest.raises(ValueError, match="nonsense"):
        _index(["nonsense"]).filesystem(TIME)


def test_several_grids_need_a_grid_label():
    with pytest.raises(ValueError, match="grid_label"):
        _index(["psl"]).filesystem(TIME)
    assert _index(["psl"], grid_label="gr").filesystem(TIME) == {"psl": "gs://test-bucket/psl/gr/v20190101/"}
    with pytest.raises(ValueError, match="psl"):
        _index(["psl"], grid_label="xx").filesystem(TIME)


def test_exists_does_not_check_the_disk():
    assert _index(["tas"]).exists(TIME) is True
    assert _index(["nonsense"]).exists(TIME) is False


def test_exists_accepts_querytime_keyword():
    assert _index(["tas"]).exists(querytime=TIME) is True
    assert _index(["nonsense"]).exists(querytime=TIME) is False


def test_get_wraps_lookup_errors():
    with pytest.raises(pyearthtools.data.DataNotFoundError):
        _index(["nonsense"]).get(TIME)


def test_load_merges_one_store_per_variable(monkeypatch):
    seen = {}
    monkeypatch.setattr(xarray, "open_zarr", _fake_open_zarr(seen))
    index = _index(["tas", "pr"])

    dataset = index.load(index.filesystem(TIME), chunks={})

    assert set(dataset.data_vars) == {"tas", "pr"}
    assert set(seen) == {TAS_URL, PR_URL}
    for kwargs in seen.values():
        assert kwargs["consolidated"] is True
        assert kwargs["storage_options"] == {"token": "anon"}
        assert kwargs["chunks"] == {}


def test_retrieve_selects_time_and_keeps_requested_variables(monkeypatch):
    monkeypatch.setattr(xarray, "open_zarr", _fake_open_zarr({}))

    dataset = _index(["tas"]).retrieve("2000-02")

    assert list(dataset.data_vars) == ["tas"]
    assert dataset["tas"].values.tolist() == [2.0]


def test_load_drops_conflicting_scalar_coordinates(monkeypatch):
    """tas (2 m) and uas (10 m) carry different scalar height coordinates, which must not stop the merge."""
    heights = {"tas": 2.0, "uas": 10.0}
    times = pandas.date_range("2000-01-01", periods=3, freq="MS")

    def fake(url, **kwargs):
        name = url.rstrip("/").split("/")[-3]
        return xarray.Dataset({name: ("time", [1.0, 2.0, 3.0])}, coords={"time": times, "height": heights[name]})

    monkeypatch.setattr(xarray, "open_zarr", fake)
    index = _index(["tas", "uas"])

    dataset = index.load(index.filesystem(TIME))

    assert set(dataset.data_vars) == {"tas", "uas"}
    assert "height" not in dataset.coords


def test_catalog_lists_available_data():
    catalog = PangeoCMIP6.catalog(CATALOG)

    assert isinstance(catalog, pandas.DataFrame)
    assert {"source_id", "experiment_id", "variable_id", "zstore"} <= set(catalog.columns)
    assert set(catalog["variable_id"]) == {"tas", "pr", "psl", "uas"}
    assert len(catalog) == 9


def test_ambiguous_runs_at_latest_version_are_rejected():
    index = _index(["tas"], source_id="AMBIG-MODEL", experiment_id="dcppA-hindcast")
    with pytest.raises(ValueError, match="several stores"):
        index.filesystem(TIME)


def test_ambiguous_runs_at_latest_version_are_rejected():
    index = _index(["tas"], source_id="AMBIG-MODEL", experiment_id="dcppA-hindcast")
    with pytest.raises(ValueError, match="several stores"):
        index.filesystem(TIME)


def test_retrieve_outside_the_store_times_raises(monkeypatch):
    monkeypatch.setattr(xarray, "open_zarr", _fake_open_zarr({}))

    with pytest.raises(pyearthtools.data.DataNotFoundError):
        _index(["tas"]).retrieve("1990-01")
