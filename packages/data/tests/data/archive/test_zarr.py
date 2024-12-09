# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.


import xarray as xr

import pyearthtools.data


def test_create(tmp_path):
    zarr_archive = pyearthtools.data.archive.ZarrIndex(tmp_path / "Test.zarr")
    fake_index = pyearthtools.data.indexes.FakeIndex()

    zarr_archive.save(fake_index["2000-01-01T00"])

    assert (tmp_path / "Test.zarr" / "data").exists()


def test_combine_two_steps(tmp_path):
    zarr_archive = pyearthtools.data.archive.ZarrIndex(tmp_path / "Test.zarr")
    fake_index = pyearthtools.data.indexes.FakeIndex()

    zarr_archive.save(fake_index["2000-01-01T00"])
    zarr_archive.save(fake_index["2000-01-01T06"], mode="sa", append_dim="time")

    assert len(zarr_archive().time.values) == 2


def test_create_template(tmp_path):
    zarr_archive = pyearthtools.data.archive.ZarrIndex(tmp_path / "Test.zarr")
    fake_index = pyearthtools.data.indexes.FakeIndex()

    zarr_archive.make_template(fake_index["2000-01-01T00"], time=[0, 1, 2, 3, 4])

    assert len(zarr_archive().time.values) == 5


def test_add_to_template(tmp_path):
    zarr_archive = pyearthtools.data.archive.ZarrTimeIndex(tmp_path / "Test.zarr", template=True)
    fake_index = pyearthtools.data.indexes.FakeIndex()

    zarr_archive.make_template(
        fake_index["2000-01-01T00"],
        time=map(lambda x: x.datetime64(), pyearthtools.data.TimeRange("2000-01-01T00", "2000-01-02T00", "6 hours")),
    )
    zarr_archive.save(fake_index["2000-01-01T00"])

    assert zarr_archive("2000-01-01T00")["data"].notnull().all()
    assert not zarr_archive("2000-01-01T06")["data"].notnull().all()


def test_combine_two_steps_exists(tmp_path):
    zarr_archive = pyearthtools.data.archive.ZarrTimeIndex(tmp_path / "Test.zarr")
    fake_index = pyearthtools.data.indexes.FakeIndex()

    zarr_archive.save(fake_index["2000-01-01T00"])
    assert zarr_archive.exists()

    zarr_archive.save(fake_index["2000-01-01T06"], mode="sa", append_dim="time")

    assert zarr_archive.exists("2000-01-01T06")
    assert not zarr_archive.exists("2000-01-01T12")


def test_combine_two_steps_time_aware(tmp_path):
    zarr_archive = pyearthtools.data.archive.ZarrTimeIndex(tmp_path / "Test.zarr")
    fake_index = pyearthtools.data.indexes.FakeIndex()

    zarr_archive.save(fake_index["2000-01-01T00"])
    zarr_archive.save(fake_index["2000-01-01T06"], mode="sa", append_dim="time")

    assert len(zarr_archive("2000-01-01T00").time.values) == 1
