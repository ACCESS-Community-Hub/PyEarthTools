# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
# Copernicus Data Store Downloaders

## Available

| Name | Description |
| ---- | ----------- |
| `root_cds` | Base class for Copernius access. `_get_from_cds` must be implemented. |
| `cds` | General Copernicus downloader, uses init args to define query. |
| `ERA5` | ERA5 specific downloader. |

"""

from pyearthtools.data.download.cds.cds import cds, root_cds
from pyearthtools.data.download.cds.ERA5 import ERA5
