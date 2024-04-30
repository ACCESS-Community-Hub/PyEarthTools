# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

"""
`edit.data`

Provide a unified way to index into and retrieve data.

At the moment, data is confined to geospatial netcdf sources.

## Examples
=== "ERA5"
    ```python
    import edit.data

    ## Date of interest
    doi = '2022-04-01T03:00'

    ## Initialise the Data Loader
    dataloader = edit.data.archive.ERA5(variables = 'tmax', level = 'single')

    ## Get Data
    dataloader(doi)

    # <xarray.Dataset>
    # Dimensions:               (time: 1, latitude: 361, longitude: 720)
    # Coordinates:
    # * longitude               (longitude) float32 -180.0 -179.5 -179.0 ... 178.5 179.0 179.5
    # * latitude                (latitude) float32 90.0 89.5 89.0 88.5 ... -89.0 -89.5 -90.0
    # * time                    (time) datetime64[ns] 2022-04-01T03:00:00
    # Data variables:
    #     tmax                  (time, latitude, longitude) float32

    ```

=== "Expanded Date Pattern"
    ```python
    import edit.data

    ## Date of interest
    doi = '2022-04-01T03:00'

    ## Initialise the Data Loader
    dataloader = edit.data.patterns.ExpandedDate(root_dir = '/data/is/here/', extension = 'nc')

    ## Find Data
    dataloader.search(doi)

    # '/data/is/here/2022/04/01/20229401T0300.nc'
    ```

=== "Geographic Files"
    ```python
    import edit.data

    ## Initialise the Data Loader
    dataloader = edit.data.static.geographic()

    ## Find Data
    dataloader('world')

    ## Shapefiles for all countries in the world
    ```
"""

SEARCH_FUNCTION = "filesystem"
WARNING_THRESHOLD = 5

EXPERIMENTAL = False

from edit.data.time import EDITDatetime, TimeResolution, TimeDelta, TimeRange
from edit.data.time import EDITDatetime as datetime

from edit.data.exceptions import DataNotFoundError, InvalidIndexError
from edit.data.warnings import IndexWarning, EDITDataWarning, AccessorRegistrationWarning


from edit.data.collection import Collection, LabelledCollection
from edit.data.catalog import Catalog, CatalogEntry

from edit.data.indexes import (
    Index,
    DataIndex,
    FileSystemIndex,
    TimeIndex,
    AdvancedTimeIndex,
    AdvancedTimeDataIndex,
    BaseTimeIndex,
    DataFileSystemIndex,
    ArchiveIndex,
    ForecastIndex,
    StaticDataIndex,
    BaseCacheIndex,
    CachingIndex,
    CachingForecastIndex,
    IntakeIndex,
    IntakeIndexCache,
)
from edit.data.indexes import register_accessor

from edit.data import operations as op

from edit.data import archive, operations, static, transform, patterns, download
from edit.data.patterns import PatternIndex

from edit.data.transform.transform import (
    Transform,
    TransformCollection,
    FunctionTransform,
)
from edit.data.transform.derive import evaluate
from edit.data import save
from edit.data.save import ManageFiles, ManageTemp

"""Auto import archives if available"""
from edit.data.archive.utils import auto_import

auto_import()

"""Config Root Directories"""
archive.config_root()

Default_Catalog = Catalog(name="Default")
load = Catalog.load


def toggle_experimental():
    global EXPERIMENTAL
    EXPERIMENTAL = not (EXPERIMENTAL)
    print("\033[101;95;1m" + f"WARNING: Experimental features are now toggled {'on' if EXPERIMENTAL else 'off'}.")


import warnings as __python_warnings

__python_warnings.warn("`edit` is under heavy development and may not continue to be supported.", UserWarning)


__version__ = "2024.03.01"
