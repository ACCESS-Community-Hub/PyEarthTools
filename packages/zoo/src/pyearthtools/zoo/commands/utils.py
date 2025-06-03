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
Command utility functions for `pyearthtools.zoo`
"""

from __future__ import annotations

from typing import Any

import click
from pyearthtools.zoo import utils


def parse_args_to_dict(*args: str) -> dict[str, Any]:
    """
    Convert a list of arguments into a dictionary.

    Of the format, [--key, value, --key2, value]

    Raises:
        KeyError:
            If cannot parse args

    Returns:
        (dict[str, Any]):
            Dictionary of args
    """
    d = {}

    def remove_surrounding(query: str, char: str = "'") -> str:
        """
        Remove prefix and suffix of char if present
        """
        return query.removeprefix(char).removesuffix(char)

    list_args = list(map(lambda x: remove_surrounding(x.strip()), args))

    while "" in list_args:
        list_args.remove("")

    if len(list_args) > 0:
        i = 0
        while i < len(list_args):
            if not str(list_args[i]).startswith("--"):
                raise KeyError(f"{args[i]} is an invalid keyword argument, ensure it starts with '--'")

            key = str(list_args[i]).replace("--", "")

            if i == (len(list_args) - 1) or (i < len(list_args) - 1 and list_args[i + 1].startswith("--")):
                d[key] = True
                i += 1
            else:
                d[key] = utils.parse_str(list_args[i + 1])
                i += 2
    return d


def get_keyword_from_ctx(ctx: click.Context) -> dict[str, Any]:
    """Convert click Context to a dictionary

    Args:
        ctx (click.Context):
            Extra click context

    Returns:
        (dict[str, Any]):
            Context as a dictionary
    """
    return parse_args_to_dict(*ctx.args)


def parse_str_to_dict(user_input: str) -> dict[str, Any]:
    """
    Parse user input string in form of click input to dictionary.

    >>> parse_str_to_dict('--variable value')
    {'variable': 'value'}

    Args:
        user_input (str):
            User input

    Returns:
        (dict[str, Any]):
            Dictionry of key:value
    """

    elements = user_input.replace("--", ";&;--").split(";&;")
    complete_split: list[str] = []
    for elem in elements:
        for sp in elem.split(" ", maxsplit=1):
            complete_split.append(sp)
    return parse_args_to_dict(*complete_split)
