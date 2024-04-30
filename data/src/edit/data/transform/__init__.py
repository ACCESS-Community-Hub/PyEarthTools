# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

"""
A collection of useful transformations to apply to a [DataIndex][edit.data.DataIndex] or just a [Dataset][xarray.Dataset].

These can be called with a Dataset or given to a data retrieval function, in which case they will be applied as soon as possible.

## Examples
### Prebuilt
#### [Region][edit.data.transform.region]
    >>> import edit.data
    >>> edit.data.transforms.region.bounding(-50, -10, 110, 155)
    Transform:
        BoundingCut                   Cut Dataset to specified Bounding Box

### Custom
For more complex Transforms, the [Transform][edit.data.transform.Transform] Class can be implemented

A user must implement the [.apply()][edit.data.transform.Transform.apply] function

It is also important to note, that these Transforms can be used independently just like a function.

``` python
import edit.data

class CustomTransform(edit.data.transform.Transform):
    "Custom Transform Class to mark the xarray dataset"
    def __init__(self, value):
        self.value = value
    def apply(self, dataset):
        dataset.attrs['Transform Mark'] = self.value
        return dataset

```

"""

from edit.data.transform.transform import Transform, TransformCollection, FunctionTransform

from edit.data.transform import (
    coordinates,
    dimensions,
    normalisation,
    utils,
    variables,
    attributes,
    optimisation,
    values,
)
from edit.data.transform.aggregation import AggregateTransform as aggregation
from edit.data.transform.default import get_default_transforms
from edit.data.transform.interpolation import InterpolateTransform as interpolation
from edit.data.transform.interpolation import interpolate_na
from edit.data.transform.mask import MaskTransform as mask
from edit.data.transform.region import RegionTransform as Region
from edit.data.transform.region import RegionTransform as region
from edit.data.transform.derive import derive
