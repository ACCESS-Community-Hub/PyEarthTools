# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""Saving and Loading of `Indexes`"""

from typing import Union

from pathlib import Path
import os

import yaml

from edit.utils import initialisation

import edit.data
from edit.data.utils import parse_path

CONFIG_KEY = "--CONFIG--"


def load(stream: Union[str, Path], **kwargs) -> "edit.data.Index":
    """
    Load a `saved` `edit.data.Index`

    Args:
        stream (Union[str, Path]):
            Stream to load, can be either path to config or yaml str

    Returns:
        (edit.data.Index):
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
