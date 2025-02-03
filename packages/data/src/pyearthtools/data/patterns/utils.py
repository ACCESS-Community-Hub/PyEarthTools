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

import tempfile
import re
import os

from pathlib import Path


def parse_root_dir(root_dir: str | Path, ignore_temp: bool = False) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    """
    Parse given `root_dir`


    If `root_dir` == 'temp', create a temporary directory, and return it

    Parse environment variables. .e.g. $USER evalutes correctly.

    Args:
        root_dir (str | Path):
            Root directory to parse.
        ignore_temp (bool, optional):
            Whether to ignore `root_dir` == 'temp.
            Defaults to False

    Returns:
        tuple[Path, tempfile.TemporaryDirectory | None]:
            Path of `root_dir`, but with it parsed and resolved, and if was temp, the associated temp directory object.

    """
    temp_dir = None

    root_dir = str(root_dir)

    if isinstance(root_dir, str) and root_dir == "temp" and not ignore_temp:
        temp_dir = tempfile.TemporaryDirectory()
        root_dir = temp_dir.name

    matches: list[str] = re.findall(r"(\$[A-z0-9]+)", root_dir)
    for match in matches:
        key = match.replace("$", "")
        if key not in os.environ:
            raise ValueError(
                f"{match} was not present in the os environment. Cannot parse {root_dir!r}.",
            )
        root_dir = root_dir.replace(match, os.environ[key])
    return Path(root_dir).resolve().absolute(), temp_dir
