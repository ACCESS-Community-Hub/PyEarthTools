# # Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# import pytest
# import sys
# import functools

# from typing import Type

# import pyearthtools.zoo


# @functools.lru_cache(None)
# def get_model() -> Type[pyearthtools.zoo.BaseForecastModel]:
#     sys.path.append(f"{__file__}/../..")
#     from fakemodel import FakeModel

#     return FakeModel


# def load_model(**kwargs) -> pyearthtools.zoo.BaseForecastModel:
#     return get_model()(**kwargs)


# def test_warn_model():
#     with pytest.warns(pyearthtools.zoo.AccessorRegistrationWarning):
#         pyearthtools.zoo.register("Testing/Warning")(lambda x: x)
#         pyearthtools.zoo.register("Testing/Warning")(lambda x: x)


# def test_available_models():
#     get_model()
#     assert "Testing/FakeModel" in pyearthtools.zoo.available_models()


# def test_name():
#     model = get_model()
#     assert model.get_name() == "Testing/FakeModel"


# def test_as_none_name(monkeypatch):
#     model = get_model()
#     monkeypatch.setattr(model, "_name", None)
#     assert model.get_name() == "FakeModel"


# def test_valid_pipelines():
#     model = get_model()
#     assert "fakedata" in model.valid_pipelines()


# @pytest.mark.parametrize(
#     "pipeline",
#     [
#         ("fakedata"),
#         ("fakedata{test=10}"),
#     ],
# )
# def test_isvalid(pipeline):
#     model = get_model()
#     assert model.is_valid_pipeline(pipeline)


# @pytest.mark.parametrize(
#     "pipeline",
#     [
#         ("fakedata"),
#         ("fakedata{test=10}"),
#     ],
# )
# def test_isvalid_with_other_config(pipeline):
#     model = get_model()
#     assert model.is_valid_pipeline(pipeline, config_path="./")


# @pytest.mark.parametrize(
#     "pipeline",
#     [
#         ("fakedata_"),
#         ("fakedata(test)"),
#         ("fakedata[test=10]"),
#     ],
# )
# def test_isnotvalid(pipeline):
#     model = get_model()
#     assert not model.is_valid_pipeline(pipeline)


# @pytest.mark.parametrize(
#     "pipeline",
#     [
#         ("fakedata_"),
#         ("fakedata(test)"),
#         ("fakedata[test=10]"),
#     ],
# )
# def test_isnotvalid_init(pipeline):
#     model = get_model()
#     with pytest.raises(ValueError):
#         model(pipeline=pipeline, output=None)

# @pytest.mark.xfail
# def test_data():
#     import tempfile
#     from pathlib import Path

#     temp_dir = tempfile.TemporaryDirectory()
#     model = load_model(pipeline="fakedata", output=temp_dir.name, data_cache=temp_dir.name)
#     model.data("2020-01-01T00")

#     assert (Path(temp_dir.name) / "Testing/FakeModel/fakedata/2020/20200101T0000.nc").exists(), "File does not exist"


# # def test_fail_to_run(monkeypatch):
# #     import tempfile

# #     temp_dir = tempfile.TemporaryDirectory()
# #     model = load_model(pipeline="fakedata", output=temp_dir.name)
# #     monkeypatch.setattr(model, "_config_path", None)
# #     monkeypatch.setattr(model, "_default_config_path", None)

# #     with pytest.raises(RuntimeError):
# #         model.run("2020-01-01T00")


# @pytest.mark.xfail
# def test_file_exists_altered_config():
#     import tempfile
#     from pathlib import Path

#     temp_dir = tempfile.TemporaryDirectory()
#     model = load_model(pipeline="fakedata", output=temp_dir.name, config_path="./")
#     model.run("2020-01-01T00")

#     assert (Path(temp_dir.name) / "data/2020/20200101T0000.nc").exists(), "File does not exist"

# @pytest.mark.xfail
# def test_file_exists():
#     import tempfile
#     from pathlib import Path

#     temp_dir = tempfile.TemporaryDirectory()
#     model = load_model(pipeline="fakedata", output=temp_dir.name)
#     model.run("2020-01-01T00")

#     assert (Path(temp_dir.name) / "data/2020/20200101T0000.nc").exists(), "File does not exist"

# @pytest.mark.xfail
# def test_file_exists_pattern_change():
#     import tempfile
#     from pathlib import Path

#     temp_dir = tempfile.TemporaryDirectory()
#     model = load_model(pipeline="fakedata", output=temp_dir.name, pattern="DirectVariable")
#     model.run("2020-01-01T00")
#     # print(list(Path(temp_dir.name).rglob('*')))
#     assert (Path(temp_dir.name) / "data/20200101T0000.nc").exists(), "File does not exist"

# @pytest.mark.xfail
# def test_search():
#     import tempfile

#     temp_dir = tempfile.TemporaryDirectory()
#     model = load_model(pipeline="fakedata", output=temp_dir.name)
#     model("2020-01-01T00")

#     assert model.search("2020-01-01T00")["data"].exists()

# @pytest.mark.xfail
# def test_max_value_config_change():
#     import tempfile
#     from pathlib import Path
#     import xarray as xr

#     temp_dir = tempfile.TemporaryDirectory()
#     model = load_model(
#         pipeline="fakedata{MaxValue=0.0}",
#         output=temp_dir.name,
#     )
#     model.run("2020-01-01T00")

#     data = xr.open_dataset(Path(temp_dir.name) / "data/2020/20200101T0000.nc")
#     assert data.max().compute().data.values == 0.0

# @pytest.mark.xfail
# def test_attributes():
#     import tempfile
#     from pathlib import Path
#     import xarray as xr

#     temp_dir = tempfile.TemporaryDirectory()
#     model = load_model(
#         pipeline="fakedata",
#         output=temp_dir.name,
#     )
#     model.run("2020-01-01T00")

#     data = xr.open_dataset(Path(temp_dir.name) / "data/2020/20200101T0000.nc")
#     assert "pyearthtools_models" in data.attrs
