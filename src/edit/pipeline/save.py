# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""Saving and Loading of `Pipelines`"""

import os
from typing import Any, Union, Optional

from pathlib import Path
import warnings

import yaml

from edit.data.utils import parse_path

from edit.utils.initialisation.imports import dynamic_import
from edit.utils import initialisation

import edit.pipeline

CONFIG_KEY = "--CONFIG--"
SUFFIX = '.pipe'


def save(pipeline: "edit.pipeline.Pipeline", path: Optional[Union[str, Path]] = None) -> Union[None, str]:
    """
    Save `Pipeline`

    Args:
        pipeline (edit.pipeline.Pipeline):
            Pipeline to save
        path (Optional[FILE], optional):
            File to save to. If not given return save str. Defaults to None.

    Returns:
        (Union[None, str]):
            If `path` is None, `pipeline` in save form else None.
    """
    pipeline_yaml = yaml.dump(pipeline, Dumper=initialisation.Dumper, sort_keys=False)

    extra_info: dict[str, Any] = {"VERSION": edit.pipeline.__version__}
    import_locations = [
        step._import for step in pipeline.flattened_steps if hasattr(step, "_import") and getattr(step, "_import")
    ]
    extra_info["import"] = import_locations

    full_yaml = pipeline_yaml + f"\n{CONFIG_KEY}\n" + yaml.dump(extra_info)

    if path is None:
        return full_yaml
    
    path = Path(parse_path(path))

    path.parent.mkdir(parents=True, exist_ok=True)
    path = path.with_suffix(SUFFIX)

    with open((path), "w") as file:
        file.write(full_yaml)


def load(stream: Union[str, Path], **kwargs: Any) -> "edit.pipeline.Pipeline":
    """
    Load `Pipeline` config

    Args:
        stream (Union[str, Path]):
            File or dump to load
        kwargs (Any):
            Updates to default values include in the config.

    Returns:
        (edit.pipeline.Pipeline):
            Loaded Pipeline
    """
    contents = None

    if os.path.sep in str(stream) or os.path.exists(parse_path(stream)):
        try:
            if Path(parse_path(stream)).is_dir():
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
    if not isinstance(loaded_obj, edit.pipeline.Pipeline):
        raise FileNotFoundError(f"Cannot load {stream!r}, is it a valid Pipeline?")
    return loaded_obj