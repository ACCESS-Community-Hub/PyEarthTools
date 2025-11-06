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


# ruff: noqa: F401
"""
`pyearthtools.data`

Provide a unified way to index into and retrieve data.

At the moment, data is confined to geospatial netcdf sources.

## Examples
=== "ERA5"
    ```python
    import pyearthtools.data

    ## Date of interest
    doi = '2022-04-01T03:00'

    ## Initialise the Data Loader
    dataloader = pyearthtools.data.archive.ERA5(variables = 'tmax')

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
    import pyearthtools.data

    ## Date of interest
    doi = '2022-04-01T03:00'

    ## Initialise the Data Loader
    dataloader = pyearthtools.data.patterns.ExpandedDate(root_dir = '/data/is/here/', extension = 'nc')

    ## Find Data
    dataloader.search(doi)

    # '/data/is/here/2022/04/01/20229401T0300.nc'
    ```

=== "Geographic Files"
    ```python
    import pyearthtools.data

    ## Initialise the Data Loader
    dataloader = pyearthtools.data.static.geographic()

    ## Find Data
    dataloader('world')

    ## Shapefiles for all countries in the world
    ```
"""

__version__ = "0.5.1"

import warnings as __python_warnings

import pyearthtools.utils
from pyearthtools.data.time import Petdt, TimeDelta, TimeRange, TimeResolution, time_delta_resolution

from pyearthtools.data import (
    archive,
    config,
    derived,
    download,
    indexes,
    logger,
    modifications,
    operations,
    patterns,
    save,
    static,
    transforms,
    utils,
)
from pyearthtools.data import transforms as transform
from pyearthtools.data.archive.utils import auto_import
from pyearthtools.data.collection import Collection, LabelledCollection
from pyearthtools.data.exceptions import DataNotFoundError, InvalidIndexError

# from pyearthtools.data.catalog import Catalog, CatalogEntry
from pyearthtools.data.indexes import (
    AdvancedTimeDataIndex,
    AdvancedTimeIndex,
    ArchiveIndex,
    BaseTimeIndex,
    CachingForecastIndex,
    CachingIndex,
    DataFileSystemIndex,
    DataIndex,
    FileSystemIndex,
    ForecastIndex,
    Index,
    IntakeIndex,
    IntakeIndexCache,
    StaticDataIndex,
    TimeIndex,
    register_accessor,
)
from pyearthtools.data.load import load
from pyearthtools.data.patterns import PatternIndex
from pyearthtools.data.save import ManageFiles, ManageTemp
from pyearthtools.data.time import Petdt as datetime
from pyearthtools.data.transforms.derive import evaluate
from pyearthtools.data.transforms.transform import (
    FunctionTransform,
    Transform,
    TransformCollection,
)
from pyearthtools.data.warnings import (
    AccessorRegistrationWarning,
    IndexWarning,
    pyearthtoolsDataWarning,
)

"""Auto import archives if available"""

auto_import()

"""Config Root Directories"""
archive.config_root()


__all__ = [
    "Petdt",
    "TimeDelta",
    "TimeRange",
    "TimeResolution",
    "evaluate",
    "FunctionTransform",
    "Transform",
    "TransformCollection",
    "IndexWarning",
    "AccessorRegistrationWarning",
    "pyearthtoolsDataWarning",
    "AdvancedTimeDataIndex",
    "AdvancedTimeIndex",
    "ArchiveIndex",
    "BaseTimeIndex",
    "CachingForecastIndex",
    "CachingIndex",
    "DataFileSystemIndex",
    "DataIndex",
    "FileSystemIndex",
    "ForecastIndex",
    "Index",
    "IntakeIndex",
    "IntakeIndexCache",
    "StaticDataIndex",
    "TimeIndex",
    "register_accessor",
    "Collection",
    "LabelledCollection",
]
