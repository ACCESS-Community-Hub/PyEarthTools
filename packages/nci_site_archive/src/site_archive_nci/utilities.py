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

"""Utilities for NCI indexes"""

from __future__ import annotations

import functools
from pathlib import Path

JOIN_LINK = "https://my.nci.org.au/mancini/project/{code}/join"


def check_project(project_code: str, scratch: bool = False) -> bool:
    """
    Check project code data existance.
    """

    default_root_path = Path("/g/data/")
    project_code = str(project_code)

    if "/scratch" in project_code or scratch:
        default_root_path = Path("/scratch/")

    project_code = project_code.replace("/g/data/", "").replace("/scratch/", "").split("/")[0]

    if not (default_root_path / project_code).exists():
        raise FileNotFoundError(
            f"Could not find data path for {project_code!r}."
            "\nTherefore no data can be loaded from this index."
            f"\nJoin this project at {JOIN_LINK.format(code = project_code)}"
        )
    return True


@functools.lru_cache()
def cached_iterdir(path: Path) -> list[Path]:
    """Run iterdir but cached"""
    return list(path.iterdir())


@functools.lru_cache()
def cached_exists(path: Path) -> bool:
    """Run exits but cached"""
    return path.exists()
