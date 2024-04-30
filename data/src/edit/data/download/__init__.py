# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

"""
Download Based Data Indexes for `edit.data`

Implemented:

| Name | Description |
| ---- | ----------- |
| `DownloadIndex`  | Base download index. `download` must be implemented. |
| `cds` | Copernicus Data Store Access |
| `opendata` | ECMWF Opendata |

"""

from edit.data.download.templates import DownloadIndex
from edit.data.download import cds
from edit.data.download import ecmwf_opendata as opendata
