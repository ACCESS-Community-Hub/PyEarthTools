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


"""Saving and Loading of `Pipelines`"""

import os
from typing import Any, Union, Optional

from pathlib import Path
import warnings

import yaml

import logging

from pyearthtools.data.utils import parse_path

from pyearthtools.utils.initialisation.imports import dynamic_import
from pyearthtools.utils import initialisation

import pyearthtools.pipeline

CONFIG_KEY = "--CONFIG--"
SUFFIX = ".epi"

LOG = logging.getLogger("pyearthtools.pipeline")


def save(pipeline: "pyearthtools.pipeline.Pipeline", path: Optional[Union[str, Path]] = None) -> Union[None, str]:
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
    pipeline_yaml = yaml.dump(pipeline, Dumper=initialisation.Dumper, sort_keys=False)

    extra_info: dict[str, Any] = {"VERSION": pyearthtools.pipeline.__version__}
    import_locations = [
        step._import for step in pipeline.flattened_steps if hasattr(step, "_import") and getattr(step, "_import")
    ]
    extra_info["import"] = import_locations

    full_yaml = pipeline_yaml + f"\n{CONFIG_KEY}\n" + yaml.dump(extra_info)

    if path is None:
        return full_yaml

    path = parse_path(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.suffix:
        path = path.with_suffix(SUFFIX)

    with open((path), "w") as file:
        file.write(full_yaml)


def load(stream: Union[str, Path], **kwargs: Any) -> "pyearthtools.pipeline.Pipeline":
    """
    Load `Pipeline` config

    Args:
        stream (Union[str, Path]):
            File or dump to load
        kwargs (Any):
            Updates to default values include in the config.

    Returns:
        (pyearthtools.pipeline.Pipeline):
            Loaded Pipeline
    """
    LOG.debug(f"Loading stream {stream}")

    contents = None

    if os.path.sep in str(stream) or parse_path(stream).exists():
        try:
            if parse_path(stream).is_dir():
                raise FileNotFoundError(f"{parse_path(stream)!r} is directory and cannot be opened.")
            contents = "".join(open(str(parse_path(stream))).readlines())
        except OSError:
            pass

    if contents is None:
        contents = str(stream)

    if not isinstance(contents, str):
        raise TypeError(f"Cannot parse contents of type {type(contents)} -{contents}.")

    contents = initialisation.update_contents(contents, **kwargs)

    if CONFIG_KEY in contents:
        config_str = contents[contents.index(CONFIG_KEY) :].replace(CONFIG_KEY, "")
        contents = contents[: contents.index(CONFIG_KEY)].replace(CONFIG_KEY, "")
        config = yaml.load(config_str, yaml.Loader)
    else:
        config = {}

    if "import" in config:
        for i in config["import"]:
            try:
                dynamic_import(i)
            except (ImportError, ModuleNotFoundError):
                warnings.warn(
                    f"Could not import {i}",
                    UserWarning,
                )

    loaded_obj = yaml.load(contents, initialisation.Loader)

    LOG.debug(f"Loaded pipeline object: {loaded_obj = }")

    if not isinstance(loaded_obj, pyearthtools.pipeline.Pipeline):
        raise FileNotFoundError(f"Cannot load {stream!r}, is it a valid Pipeline?")
    return loaded_obj
