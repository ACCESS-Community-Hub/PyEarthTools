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


"""
Utilities to delete files
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import time
from typing import Literal, Sequence
import logging

from pyearthtools.data import TimeDelta
from pyearthtools.utils.context import Catch

LOG = logging.getLogger("pyearthtools.data")


def delete_path(
    path: os.PathLike | list[os.PathLike] | tuple[os.PathLike, ...] | dict[str, os.PathLike],
    remove_empty_dirs: bool = False,
):
    """Delete all paths"""

    if isinstance(path, dict):
        list([delete_path(value) for value in path.values()])
        return

    elif isinstance(path, (tuple, list)):
        list([delete_path(value) for value in path])
        return

    elif isinstance(path, (str, Path)):
        path = Path(path)

        if not path.exists():
            return

        elif path.exists() and path.is_dir():
            shutil.rmtree(path)

        elif path.exists() and path.is_file():
            with Catch(FileNotFoundError):
                os.remove(str(path))

        if remove_empty_dirs and len(list(path.parent.glob("*"))) == 0:
            delete_path(path.parent, remove_empty_dirs=remove_empty_dirs)
        return

    raise TypeError(f"Cannot parse path of type: {type(path)!r}")


def delete_older_than(
    paths: Sequence[os.PathLike],
    delta: TimeDelta | str,
    key: Literal["modified", "created"] = "modified",
    verbose: bool = False,
    remove_empty_dirs: bool = False,
):
    """Delete all paths older than delta"""

    key_to_func = {
        "modified": os.path.getmtime,
        "created": os.path.getctime,
    }
    func = key_to_func[key]

    for path in paths:
        if not Path(path).exists():
            continue

        if (time.time() - func(path)) > TimeDelta(delta).total_seconds():
            msg = f"Deleting '{path}' as it is older than {delta}'s."
            LOG.debug(msg)
            if verbose:
                print("\033[41;1;2m" + msg)
            delete_path(path, remove_empty_dirs=remove_empty_dirs)
