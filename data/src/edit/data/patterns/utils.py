# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

from __future__ import annotations

import tempfile
import re
import os

from pathlib import Path


def parse_root_dir(root_dir: str | Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    """
    Parse given `root_dir`


    If `root_dir` == 'temp', create a temporary directory, and return it

    Parse environment variables. .e.g. $USER evalutes correctly.

    Args:
        root_dir (str | Path):
            Root directory to parse.

    Returns:
        tuple[Path, tempfile.TemporaryDirectory | None]:
            Path of `root_dir`, but with it parsed and resolved, and if was temp, the associated temp directory object.

    """
    temp_dir = None

    root_dir = str(root_dir)

    if isinstance(root_dir, str) and root_dir == "temp":
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
