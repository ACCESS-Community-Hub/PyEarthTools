# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from edit.data.indexes import FileSystemIndex


def save(data: dict, callback: FileSystemIndex, *args, save_kwargs: dict[str, Any] = {}, **kwargs):
    """Save json files"""
    path = callback.search(*args, **kwargs)
    if not isinstance(path, (str, Path)):
        raise NotImplementedError(f"Cannot handle saving with paths as {type(path)}")
    path = Path(path)

    with open(path, "w") as file:
        json.dump(data, file, indent=4, **save_kwargs)
    return path
