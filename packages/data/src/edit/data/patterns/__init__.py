# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

# ruff: noqa: F401

"""
[DataIndexes][edit.data.DataIndex] with data discovered through patterns

## Implemented

### Temporal, Variable and Normal
These patterns have temporal and variable aware versions available.
For the extra versions, either add `Variable` to the end or `Temporal` to the start.

| Name        | Description |
| :---        |     ----:   |
| [ExpandedDate][edit.data.patterns.expanded_date.ExpandedDate]  |  Time expansion based filename    |
| [Direct][edit.data.patterns.direct.Direct]  |   Direct Time based Filename      |


### Other
These patterns stand alone

| Name        | Description |
| :---        |     ----:   |
| [Argument][edit.data.patterns.argument.Argument]  |  Argument as Filename      |
| [ArgumentExpansion][edit.data.patterns.argument.ArgumentExpansion]  |  Argument Expansion Filename      |
| [Static][edit.data.patterns.static.Static]  |  Single Static File     |
| [ParsingPattern][edit.data.patterns.parser.ParsingPattern]  |  F string based parser   |

## Examples
Each Pattern has it's own examples, but here is one

```python
pattern = edit.data.patterns.ArgumentExpansion('/dir/', '.nc')
str(pattern.search('test','arg'))
# '/dir/arg/test.nc'

```
"""

from edit.data.patterns import utils

from edit.data.patterns.default import (
    PatternIndex,
    PatternTimeIndex,
    PatternForecastIndex,
    PatternVariableAware,
)

from edit.data.patterns.argument import (
    Argument,
    ArgumentExpansion,
    ArgumentExpansionVariable,
    ArgumentExpansionFactory,
)
from edit.data.patterns.direct import (
    Direct,
    TemporalDirect,
    ForecastDirect,
    DirectVariable,
    ForecastDirectVariable,
    TemporalDirectVariable,
    DirectFactory,
)
from edit.data.patterns.expanded_date import (
    ExpandedDate,
    TemporalExpandedDate,
    ForecastExpandedDate,
    ExpandedDateVariable,
    ForecastExpandedDateVariable,
    TemporalExpandedDateVariable,
    ExpandedDateFactory,
)
from edit.data.patterns.static import Static
from edit.data.patterns.parser import ParsingPattern


ZARR_IMPORTED = True
try:
    from edit.data.patterns.zarr import ZarrIndex, ZarrTimeIndex  # noqa: F401
except (ImportError, ModuleNotFoundError):
    ZARR_IMPORTED = False
