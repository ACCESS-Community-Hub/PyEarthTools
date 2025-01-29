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
[DataIndexes][pyearthtools.data.DataIndex] with data discovered through patterns

## Implemented

### Temporal, Variable and Normal
These patterns have temporal and variable aware versions available.
For the extra versions, either add `Variable` to the end or `Temporal` to the start.

| Name        | Description |
| :---        |     ----:   |
| [ExpandedDate][pyearthtools.data.patterns.expanded_date.ExpandedDate]  |  Time expansion based filename    |
| [Direct][pyearthtools.data.patterns.direct.Direct]  |   Direct Time based Filename      |


### Other
These patterns stand alone

| Name        | Description |
| :---        |     ----:   |
| [Argument][pyearthtools.data.patterns.argument.Argument]  |  Argument as Filename      |
| [ArgumentExpansion][pyearthtools.data.patterns.argument.ArgumentExpansion]  |  Argument Expansion Filename      |
| [Static][pyearthtools.data.patterns.static.Static]  |  Single Static File     |
| [ParsingPattern][pyearthtools.data.patterns.parser.ParsingPattern]  |  F string based parser   |

## Examples
Each Pattern has it's own examples, but here is one

```python
pattern = pyearthtools.data.patterns.ArgumentExpansion('/dir/', '.nc')
str(pattern.search('test','arg'))
# '/dir/arg/test.nc'

```
"""

from pyearthtools.data.patterns import utils

from pyearthtools.data.patterns.default import (
    PatternIndex,
    PatternTimeIndex,
    PatternForecastIndex,
    PatternVariableAware,
)

from pyearthtools.data.patterns.argument import (
    Argument,
    ArgumentExpansion,
    ArgumentExpansionVariable,
    ArgumentExpansionFactory,
)
from pyearthtools.data.patterns.direct import (
    Direct,
    TemporalDirect,
    ForecastDirect,
    DirectVariable,
    ForecastDirectVariable,
    TemporalDirectVariable,
    DirectFactory,
)
from pyearthtools.data.patterns.expanded_date import (
    ExpandedDate,
    TemporalExpandedDate,
    ForecastExpandedDate,
    ExpandedDateVariable,
    ForecastExpandedDateVariable,
    TemporalExpandedDateVariable,
    ExpandedDateFactory,
)
from pyearthtools.data.patterns.static import Static
from pyearthtools.data.patterns.parser import ParsingPattern


ZARR_IMPORTED = True
try:
    from pyearthtools.data.patterns.zarr import ZarrIndex, ZarrTimeIndex  # noqa: F401
except (ImportError, ModuleNotFoundError):
    ZARR_IMPORTED = False
