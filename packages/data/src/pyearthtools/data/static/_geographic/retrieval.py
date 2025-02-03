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
Allow Retrieval of Data sources from subdirectories
"""

from __future__ import annotations

import logging
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Literal

import yaml

from pyearthtools.data.exceptions import DataNotFoundError, InvalidIndexError
from pyearthtools.data.static._geographic import DATA_BASEDIRECTORY, DOWNLOAD_DATA

DOWNLOAD_EXTENSIONS = [".yaml"]
SUPPORTED_EXTENSIONS = [".shp", ".dbf", ".csv", ".xml", ".shx"]
VALID_EXTENSIONS = Literal[".shp", ".dbf", ".csv", ".xml", ".shx"]

PATH_SEPERATOR = "/"

VALID_DATA = None


def generate_key_path_pairs(
    search_directory: str | Path = DATA_BASEDIRECTORY,
    base_directory: str | Path = DATA_BASEDIRECTORY,
    ignore_download: bool = not DOWNLOAD_DATA,
) -> dict[str, list[str]]:
    """
    Generate key path pairs from data in given directory

    Args:
        search_directory (str | Path, optional):
            Directory to search for data in. Defaults to DATA_BASEDIRECTORY.
        base_directory (str | Path, optional):
            Directory to remove when creating keys. Defaults to DATA_BASEDIRECTORY.
        ignore_download (bool, optional):
            Whether to ignore download configs. Defaults to notDOWNLOAD_DATA.

    Returns:
        dict[str, list[str]]:
            Dictionary assigning keys to files
    """

    data_files: dict[str, list[str]] = {}

    for path in Path(search_directory).rglob("*"):
        path = path.resolve()
        extension = path.suffix
        foldername = path.parent

        if extension in SUPPORTED_EXTENSIONS or (extension in DOWNLOAD_EXTENSIONS and not ignore_download):
            key_path = foldername.relative_to(Path(base_directory))

            key = PATH_SEPERATOR.join(key_path.parts).lower()

            if key in data_files:
                data_files[key].append(str(path))
            else:
                data_files[key] = [str(path)]

    return data_files


def download(config_path: str | Path) -> dict[str, list[str]]:
    """From specified download config download data, unzipping if needed

    Args:
      config_path (str | Path): Location of download config

    Specifications for config file:
        url: Location to download data from
        save_name: Where to save data
        zip: If data downloads as a zip
        subfolder: optional: Name of sub directory in which data is found

    Returns:
        dict[str, list[str]]: Dictionary assigning keys to files
    """

    directory_name = Path(config_path).parent
    logging.info(f"Downloading data at {config_path}")

    try:
        # Open Download Config File
        with open(config_path, "r") as stream:
            download_dict = yaml.safe_load(stream)

        # Save Data in parent directory at specified name
        save_path = os.path.join(directory_name, download_dict["save_name"])
        url = str(download_dict["url"]).strip().replace(" ", "")
        urllib.request.urlretrieve(url, save_path)

        # Unzip if required
        if download_dict["zip"] or download_dict["zip"] == "True":
            with zipfile.ZipFile(save_path, "r") as zip_ref:
                zip_ref.extractall(os.path.dirname(save_path))
            os.remove(save_path)

        # If data is in a subfolder, pull data into same directory as download config
        if "subfolder" in download_dict:
            for filepath in (directory_name / Path(download_dict["subfolder"])).glob("*"):
                filename = os.path.basename(filepath)
                os.rename(
                    filepath,
                    os.path.join(directory_name, download_dict["subfolder"], filename),
                )
    # Except if key error,
    except KeyError as excep:
        raise KeyError(f"Download Config is missing key: '{excep.args[0]}'")

    return generate_key_path_pairs(directory_name, ignore_download=True)


def get(
    searchkey: str,
    extension: VALID_EXTENSIONS | str | None = None,
    *,
    downloaded: bool = False,
) -> str:
    """
    Retrieve Data from Australian Geographic Data based on key

    Args:
        searchkey (str): Key to search for
        extension (VALID_EXTENSIONS | str | None, optional): File extension to load. Defaults to None.

    Raises:
        InvalidIndexError: If Extension not found
        DataNotFoundError: If Key not found

    Returns:
        str: File path if file found
    """
    global VALID_DATA
    searchkey = searchkey.lower()

    if isinstance(extension, str):
        extension = f".{extension.removeprefix('.')}"

    if VALID_DATA is None:
        VALID_DATA = generate_key_path_pairs()

    # Simple Check, if key is found
    if searchkey in VALID_DATA:
        # Search for non download files first
        download_file = None

        for discovered_file in VALID_DATA[searchkey]:
            discovered_file_extension = Path(discovered_file).suffix

            if discovered_file_extension in SUPPORTED_EXTENSIONS:  # Check if extension is in supported list
                if extension is None or extension == discovered_file_extension:  # Filter by search_extension if needed
                    return discovered_file

            elif discovered_file_extension in DOWNLOAD_EXTENSIONS and not downloaded:  # Else add download files
                download_file = discovered_file

        if download_file is not None:  # Else if download found, download and add keys
            for key, value in download(download_file).items():
                if key in VALID_DATA:
                    for item in value:
                        VALID_DATA[key].append(item)
                else:
                    VALID_DATA[key] = value
            return get(searchkey, extension, downloaded=True)

        else:
            # If none found and no file to download
            # Find other valid extensions
            valid_extensions = [Path(disc_file).suffix for disc_file in VALID_DATA[searchkey]]
            for exten in valid_extensions:
                if exten in DOWNLOAD_EXTENSIONS:
                    valid_extensions.remove(exten)

            raise InvalidIndexError(
                f"Unable to find file with extension '{extension}'. with {searchkey}. "
                f"Discovered extensions are: {list(set(valid_extensions))}"
            )

    else:
        # Else descend down through components of key, looking for a download
        search_split = searchkey.split("_")
        for i in range(1, len(search_split)):
            new_key = PATH_SEPERATOR.join(search_split[:i])

            if new_key in VALID_DATA:
                for discovered_file in VALID_DATA[searchkey]:
                    extension = Path(discovered_file).suffix
                    # If Found download
                    if extension in DOWNLOAD_EXTENSIONS:
                        # Add to Valid Data Sources
                        for key, value in download(discovered_file).items():
                            if key in VALID_DATA:
                                for item in value:
                                    VALID_DATA[key].append(item)
                            else:
                                VALID_DATA[key] = value
        # Check again
        if not downloaded:
            return get(searchkey, extension, downloaded=True)

    # Otherwise raise error
    raise DataNotFoundError(
        (f"{searchkey!r} not found in Australian Geographic Data. " f"Valid Keys: {list(VALID_DATA.keys())}")
    )
