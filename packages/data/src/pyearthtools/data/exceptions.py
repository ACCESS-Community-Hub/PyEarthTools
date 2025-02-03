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
`pyearthtools.data` Exceptions
"""

from __future__ import annotations
from typing import Any, Callable

import warnings


class InvalidIndexError(KeyError):
    """
    If an invalid index was provided
    """

    def __init__(self, message, *args):
        self.message = message
        for arg in args:
            self.message += str(arg)

    def __str__(self):
        return self.message


class InvalidDataError(KeyError):
    """
    If data cannot be loaded
    """

    def __init__(self, message, *args):
        self.message = message
        for arg in args:
            self.message += str(arg)

    def __str__(self):
        return self.message


class DataNotFoundError(FileNotFoundError):
    """
    If Data was not found
    """

    def __init__(self, message, *args):
        self.message = message
        for arg in args:
            self.message += str(arg)

    def __str__(self):
        return self.message


def run_and_catch_exception(
    command: Callable,
    *args,
    exception: BaseException | tuple[BaseException] = KeyboardInterrupt,  # type: ignore
    **kwargs,
) -> Any:
    """Run a command, and catch exceptions to gracefully terminate.

    Args:
        command (Callable):
            Command to run
        exception (BaseException | tuple[BaseException], optional):
            Exception types to catch. Defaults to KeyboardInterrupt.
        *args (Any, optional):
            Arguments to pass to the command
        **kwargs (Any, optional):
            Keyword Arguments to pass to the command

    Returns:
        (Any):
            Result of command. Will return None if error caught
    """
    try:
        return command(*args, **kwargs)
    except exception as e:  # type: ignore
        warnings.warn(f"Caught {type(e)}. Attempting graceful termination...")
    return None
