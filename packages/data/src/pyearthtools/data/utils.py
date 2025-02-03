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

from typing import Any

import re
import os

from pathlib import Path
import logging

LOG = logging.getLogger("pyearthtools.data")


def parse_path(path: os.PathLike[Any] | str) -> Path:
    """
    Parse given `root_dir`

    Parse environment variables. .e.g. $USER evalutes correctly.

    Args:
        path (os.PathLike[Any]):
            Path to parse.

    Returns:
        Path:
            Parsed path

    """
    path_str = str(path)

    if path_str == "temp":
        LOG.warn(
            "Path being parsed was 'temp', yet this parser does not autocreate temp directories. Use `patterns` to use auto temp dirs."
        )

    matches: list[str] = re.findall(r"(\$[A-z0-9]+)", path_str)  # type: ignore
    for match in matches:
        key = match.replace("$", "")
        if key not in os.environ:
            raise ValueError(
                f"{match} was not present in the os environment. Cannot parse {path_str!r}.",
            )
        path_str = path_str.replace(match, os.environ[key])
    return Path(path_str).expanduser().resolve().absolute()
