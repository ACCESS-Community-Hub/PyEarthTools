# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

from __future__ import annotations

from pathlib import Path
import os
from typing import Any

import tqdm.auto as tqdm


def filter_blacklisted(names: list[str], blacklisted: list[str]) -> list[str]:
    """
    Remove `blacklisted` elements from `names`
    """
    filtered: list[str] = []
    for name in names:
        if name not in blacklisted:
            filtered.append(name)
    return filtered


def get_structure(top: str | Path, blacklisted: list[str], verbose: bool = False) -> dict[str, Any]:
    """
    Get path structure, removing `blacklisted` entries
    """
    top = Path(top)
    walker = os.walk(top)

    structure: dict[str, Any] = {}

    for dirpath, dirnames, _ in tqdm.tqdm(walker, disable=not verbose):
        sub_dict = structure
        for component in Path(dirpath).relative_to(top).parts:
            if component in blacklisted:
                continue
            if not component in sub_dict:
                sub_dict[component] = {} if filter_blacklisted(dirnames, blacklisted) else None
            sub_dict = sub_dict[component]
    return structure


def clean_structure(dictionary: dict) -> dict | list:
    """
    Clean a structure dictionary,

    Will collapse any entries without subfolders
    """
    all_None = True
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = clean_structure(value)
        if value is not None:
            all_None = False

    if all_None:
        return list(dictionary.keys())
    return dictionary


def structure(top: str | Path, blacklisted: list[str] = [], verbose: bool = False) -> dict[str, dict | list | str]:
    """Construct a file structure as a descending dictionary.

    Any `blacklisted` folders will be ignored

    If a folder's subfolders have no subfolders beneath it, that entry is
    a list representative of the subfolders of the first folder.

    However, if another folder of the same level as subfolders, any folder
    without subfolders recieves a None.

    !!! Example
        Consider the directory structure:
        ```
        root_dir
            sub_directory_1
                look_imma_folder
                me_too
            sub_directory_2
        ```
        The resulting structure would be:
        ```python
        {'root_dir': {'sub_directory_1': ['look_imma_folder', 'me_too'], 'sub_directory_2': None}}
        ```

    Args:
        top (str | Path):
            Root path to begin structure at
        blacklisted (list[str], optional):
            Blacklisted folder names to exclude. Defaults to [].
        verbose (bool, optional):
            Print while creating. Defaults to False.

    Returns:
        (dict[str, dict | list | str]):
            Structure dictionary, as a descending dictionary.


    """
    return clean_structure(get_structure(top, blacklisted, verbose=verbose))
