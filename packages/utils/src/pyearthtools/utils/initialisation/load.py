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


"""Loading utils"""

import os
from typing import Any, Union, Optional

from pathlib import Path
import re

import logging
import yaml

from pyearthtools.utils.initialisation.yaml import Dumper, Loader

LOG = logging.getLogger("pyearthtools.utils")


def try_to_number(value: Any):
    try:
        return int(value)
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        pass
    return value


def update_contents(contents: str, **kwargs) -> str:
    """
    Update contents

    Looking for str values to attempt a str replacement defined by the `kwargs`
        A key inside the dictionary must be of form `__KEY__`, with KEY being a str.

    If ':' follows the KEY part and still within '__*__', anything following will be considered the default value.

    """

    for replace in re.findall(r"__(.+)__", contents):
        default_value = None
        key = replace

        if ":" in replace:
            key, default_value = replace.split(":")

        if key not in kwargs and default_value is None:
            LOG.warn(
                f"Configuration contains what looks to be a replace value: '__{key}__', but has not been replaced. This may cause issues."
            )
            continue

        replacement_value = kwargs.get(key, None)
        if replacement_value is None:
            replacement_value = default_value
        replacement_value = try_to_number(replacement_value)

        LOG.debug(f"Replacing {replace!r} with {replacement_value}.")

        contents = contents.replace(f"__{replace}__", str(replacement_value))

    return contents


def load(stream: Union[str, Path], **kwargs):
    """
    Load pyearthtools file replacing defaults
    """
    contents = None

    if os.path.sep in str(stream) or os.path.exists(stream):
        try:
            if Path(stream).is_dir():
                raise FileNotFoundError(f"{stream!r} is directory and cannot be opened.")
            contents = "".join(open(str(stream)).readlines())
        except OSError:
            pass

    if contents is None:
        contents = str(stream)

    if not isinstance(contents, str):
        raise TypeError(f"Cannot parse contents of type {type(contents)} -{contents}.")

    contents = update_contents(contents, **kwargs)

    return yaml.load(contents, Loader)


def save(obj, path: Optional[Union[str, Path]] = None):
    """
    Save pyearthtools objects
    """
    stream = open(str(path), "w") if path else None
    return yaml.dump(obj, stream, Dumper=Dumper, sort_keys=False)  # type: ignore
