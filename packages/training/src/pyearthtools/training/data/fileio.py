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



from __future__ import annotations

from typing import Optional, Any
import os

from pathlib import Path
import yaml

import pyearthtools.pipeline
from pyearthtools.utils.initialisation import Dumper, Loader, update_contents
from pyearthtools.training.data.datamodule import PipelineDataModule

CONFIG_KEY = "--CONFIG--"
SUFFIX = ".edm"


def save(datamodule: PipelineDataModule, path: Optional[str | Path] = None) -> None | str:
    """
    Save `Pipeline`

    Args:
        pipeline (pyearthtools.pipeline.Pipeline):
            Pipeline to save
        path (Optional[FILE], optional):
            File to save to. If not given return save str. Defaults to None.

    Returns:
        (Union[None, str]):
            If `path` is None, `pipeline` in save form else None.
    """
    datamodule_yaml = yaml.dump(datamodule, Dumper=Dumper, sort_keys=False)

    extra_info: dict[str, Any] = {"VERSION": pyearthtools.pipeline.__version__}

    full_yaml = datamodule_yaml + f"\n{CONFIG_KEY}\n" + yaml.dump(extra_info)

    if path is None:
        return full_yaml

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    path = path.with_suffix(SUFFIX)

    with open(str(path), "w") as file:
        file.write(full_yaml)


def load(stream: str | Path, **kwargs: Any) -> PipelineDataModule:
    """
    Load `Datamodule` config

    Args:
        stream (Union[str, Path]):
            File or dump to load
        kwargs (Any):
            Updates to default values include in the config.

    Returns:
        (pyearthtools.pipeline.Pipeline):
            Loaded Pipeline
    """
    contents = None

    if os.path.sep in str(stream) or os.path.exists(stream):
        try:
            if Path(stream).is_dir():
                raise FileNotFoundError(f"{stream!r} is directory and cannot be opened.")
            contents = "".join(open(str(stream), "r").readlines())
        except OSError:
            pass

    if contents is None:
        contents = str(stream)

    if not isinstance(contents, str):
        raise TypeError(f"Cannot parse contents of type {type(contents)} -{contents}.")

    contents = update_contents(contents, **kwargs)

    if CONFIG_KEY in contents:
        config_str = contents[contents.index(CONFIG_KEY) :].replace(CONFIG_KEY, "")
        contents = contents[: contents.index(CONFIG_KEY)].replace(CONFIG_KEY, "")
        config = yaml.load(config_str, yaml.Loader)
    else:
        config = {}  # noqa: F841

    loaded_obj = yaml.load(contents, Loader)
    if not isinstance(loaded_obj, PipelineDataModule):
        raise FileNotFoundError(f"Cannot load {stream!r}, is it a valid datamodule?")
    return loaded_obj
