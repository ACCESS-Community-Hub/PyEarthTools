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
# Modifications 

The usage of `@variable_modifications` allows for variables to be dynamically modified when retrieved, with a particular focus on accumulation and other aggregations.

## Definitions

The distinction between `Transforms` & `Modifications` can be considered as follows

- A `Transform` operates on a given data object seperate to its source. It introduces no new information, but simply 'transforms' the information already contained. For example, a region subset, or even an aggregation over the spatial dimensions.
- A `Modification` operates on a data object with access to its source. It can thereby introduce new information, and modifies the meaning of a variable. For example, altering `tcwv` to be the accumulated `tcwv` over a length of time.

## Usage

By decorating an archive with `@variable_modifications()` modifications can be applied on a variable level.

```python
class Archive(ArchiveIndex):
    @variable_modifications(variable_keyword = 'variable')
    def __init__(self, variable):
        ...
```

When specifying the variables, the modification syntax can be used

```!MODIFICATION[KWARGS]:VARIABLE>NEWNAME```

Where, `MODIFICATION` references the modification name, `KWARGS` contains the parameters needed for the modification in json form, and `VARIABLE` is the variable name as would be used normally and with anything after `>` being the new name, which can be ommited.

Or dictionary with following keys

```text
    - source_var (REQUIRED)     Variable to modify
    - modification (REQUIRED)   Modification to apply
    - target_var                Rename of variable
    - **                        Any other keys for `modification`
```

```python
Archive('!accumulate[period = "6 hourly"]:tcwv)
```

or

```python
Archive(variable = ['!accumulate[period = "6 hourly"]:tcwv', '2t'])
```

or

```python
Archive(variable = {'source_var':'tcwv','modification':'accumulate', 'target_var':'accum_tcwv', 'period':"6 hours"})
```

## Custom

A user can provide custom modifications accessible in the exact same way to the user,

Using `@register_modification` around a Modification subclass, with the only arg being the name accessible underneath,

```python
@register_modification("aggregate")
class AggregationGeneral(Aggregation):
    ...
```

`single` and `series` will need to be implemented by the user containing the functionality to calculate the modification for single time and a series of times dataset's.


`single` takes a single timestep and expects a dataset to be returned with the variable as modified.

`series` takes a start, end and interval, as can be parsed by `pyearthtools.data.TimeRange`, and expects 
a dataset to be returned with the variable as modified but all timesteps as defined by the range.

`variable` contains the variable being modified.

`data` contains the `TimeDataIndex` to retrieve the data from.

`attribute_update` can be overridden to specify a dictionary to update the attributes with.


## Available

- constant

- hourly
- daily
- monthly

- aggregate 
- mean
- accumulate
"""


from pyearthtools.data.modifications.modification import Modification

from pyearthtools.data.modifications.register import register_modification
from pyearthtools.data.modifications.decorator import variable_modifications

from pyearthtools.data.modifications import aggregations, reductions, constants


__all__ = ["register_modification", "variable_modifications", "Modification"]
