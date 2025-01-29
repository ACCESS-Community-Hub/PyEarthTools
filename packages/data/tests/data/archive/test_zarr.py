# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



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
