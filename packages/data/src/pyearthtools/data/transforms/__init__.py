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
A collection of useful transformations to apply to a [DataIndex][pyearthtools.data.DataIndex] or just a [Dataset][xarray.Dataset].

These can be called with a Dataset or given to a data retrieval function, in which case they will be applied as soon as possible.

## Examples
### Prebuilt
#### [Region][pyearthtools.data.transforms.region]
    >>> import pyearthtools.data
    >>> pyearthtools.data.transforms.region.Bounding(-50, -10, 110, 155)
    Transform:
        BoundingCut                   Cut Dataset to specified Bounding Box

### Custom
For more complex Transforms, the [Transform][pyearthtools.data.transforms.Transform] Class can be implemented

A user must implement the [.apply()][pyearthtools.data.transforms.Transform.apply] function

It is also important to note, that these Transforms can be used independently just like a function.

``` python
import pyearthtools.data

class CustomTransform(pyearthtools.data.transforms.Transform):
    "Custom Transform Class to mark the xarray dataset"
    def __init__(self, value):
        self.value = value
    def apply(self, dataset):
        dataset.attrs['Transform Mark'] = self.value
        return dataset

```

"""

from pyearthtools.data.transforms.transform import (
    Transform,
    TransformCollection,
    FunctionTransform,
)

from pyearthtools.data.transforms import (
    aggregation,
    coordinates,
    dimensions,
    normalisation,
    utils,
    variables,
    attributes,
    optimisation,
    values,
    interpolation,
    region,
    mask,
)
from pyearthtools.data.transforms.default import get_default_transforms

# from pyearthtools.data.transforms.mask import MaskTransform as mask
from pyearthtools.data.transforms.derive import derive, Derive
