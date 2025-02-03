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


"""Saving and Loading of `Indexes`"""

from typing import Union

from pathlib import Path
import os

import yaml

from pyearthtools.utils import initialisation

import pyearthtools.data
from pyearthtools.data.utils import parse_path

CONFIG_KEY = "--CONFIG--"


def load(stream: Union[str, Path], **kwargs) -> "pyearthtools.data.Index":
    """
    Load a `saved` `pyearthtools.data.Index`

    Args:
        stream (Union[str, Path]):
            Stream to load, can be either path to config or yaml str

    Returns:
        (pyearthtools.data.Index):
            Loaded Index
    """
    contents = None

    if os.path.sep in str(stream):
        try:
            if parse_path(stream).is_dir():
                stream = list(
                    [
                        *Path(stream).glob("catalog.cat"),
                        *Path(stream).glob("catalog.edi"),
                        *Path(stream).glob("*.cat"),
                        *Path(stream).glob("*.edi"),
                    ]
                )[
                    0
                ]  # Find default save file of index

            contents = "".join(open(str(parse_path(stream))).readlines())
        except FileNotFoundError as e:
            raise e
        except OSError:
            pass
        except IndexError:
            raise FileNotFoundError(f"No default catalog could be found at {stream!r}.")

    if contents is None:
        contents = str(stream)

    if not isinstance(contents, str):
        raise TypeError(f"Cannot parse contents of type {type(contents)} - {contents}.")

    contents = initialisation.update_contents(contents, **kwargs)

    return yaml.load(str(contents), initialisation.Loader)
