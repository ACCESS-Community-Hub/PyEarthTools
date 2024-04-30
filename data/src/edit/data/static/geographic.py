# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

try:
    import geopandas as gpd

    GEOPANDAS_IMPORTED = True
except ImportError:
    GEOPANDAS_IMPORTED = False

from edit.data.indexes import FileSystemIndex
from edit.data.static import _geographic

LOADING_FUNCTIONS = {".csv": pd.read_csv, ".xml": pd.read_xml}
if GEOPANDAS_IMPORTED:
    LOADING_FUNCTIONS[".dbf"] = gpd.read_file
    LOADING_FUNCTIONS[".shp"] = gpd.read_file


class GeographicIndex(FileSystemIndex):
    def __init__(
        self,
        *,
        extension: str | None = None,
        loading_function: Callable | None = None,
    ):
        """
        Load Geographical Static Data from `edit.data.static._geographic`

        Automatically identify the correct loading function if no extension is specified

        Notes:
            If loading_function is supplied without an extension, there is no certainty what file type will be passed through

        Args:
            extension (str | None, optional):
                Particular file extension to find. Defaults to None.
            loading_function (Callable | None, optional):
                Override for the function to load the data with. Defaults to None.

        Raises:
            KeyError: If extension not recognised
        """
        super().__init__()
        self.make_catalog()

        if extension:
            extension = f".{extension.removeprefix('.')}"

        if extension is not None and extension not in LOADING_FUNCTIONS and loading_function is None:
            raise KeyError(
                f"Unable to load {extension} with known loading functions {list(LOADING_FUNCTIONS.keys())}. Try providing 'loading_function' or setting `extension`"
            )

        self.extension = extension
        self.loading_function = loading_function

    def load(
        self,
        files: str | Path,
    ) -> Any:
        """
        Load Geographical Static Data from [_geographic][edit.data.static._geographic]

        Args:
            files (str | Path):
                files to load

        Raises:
            KeyError: If key not found

        Returns:
            Any: Loaded data
        """

        files = Path(files)
        extension = self.extension or files.suffix

        load_func = self.loading_function or LOADING_FUNCTIONS[extension]
        return load_func(files)

    def search(self, searchkey: str) -> Path:
        return Path(_geographic.get(searchkey=searchkey, extension=self.extension))

    def filesystem(self, *args) -> Path | dict[str, str]:
        return self.search(*args)

    @staticmethod
    def _download_all(verbose: bool = False) -> bool:
        """
        Ensure all keys are downloaded for geographic

        Args:
            verbose (bool, optional): Print progress messages. Defaults to False.

        Returns:
            bool: Success Flag
        """
        for key, _ in _geographic.retrieval.generate_key_path_pairs().items():
            if verbose:
                print(f"Downloading: \t {key}")
            _geographic.get(key)
        return True
