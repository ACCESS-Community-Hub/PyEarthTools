# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
Geographic Data Retrieval

Allow retrieval and download of known Geographic Datasets, which are specified in config files.

Will attempt to automatically load into geopandas if installed, or simply return.
"""

from pathlib import Path

DOWNLOAD_DATA: bool = True
DATA_BASEDIRECTORY: Path = Path(__file__).parent.resolve().absolute()

from edit.data.static._geographic.retrieval import get  # noqa: E402 F401
from edit.data.static._geographic import retrieval
