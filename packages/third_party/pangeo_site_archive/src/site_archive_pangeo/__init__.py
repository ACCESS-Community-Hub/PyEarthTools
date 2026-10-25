"""
Pangeo CMIP6 archive for PyEarthTools.

| Name                                            | Description                                         |
| :---                                            | ----:                                               |
| [PangeoCMIP6][site_archive_pangeo.PangeoCMIP6]  | CMIP6 analysis-ready, cloud-optimised (ARCO) data   |
"""

from pyearthtools.data.archive import register_archive

import site_archive_pangeo
from site_archive_pangeo.pangeo_cmip6 import PangeoCMIP6

register_archive("pangeo")(site_archive_pangeo)

try:
    from ._version import __version__  # type: ignore
except ImportError:
    __version__ = "999"

__all__ = ["PangeoCMIP6"]
