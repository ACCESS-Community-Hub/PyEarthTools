# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
A sophisticated [Transform][pyearthtools.data.transforms.Transform] to normalise and unnormalise data.

## Methods
| Name        | Description |
| :---        |     ----:   |
| none      | No Normalisation |
| function | User provided function Normalisation |
| log      | Log Data |
| anomaly  | Subtract Temporal Mean |
| deviation | Subtract mean and divide by std |
| range | Find range and force between 0 & 1 |


## Transforms
[Normalise][pyearthtools.data.transforms.normalisation.normalise] provides the Transforms to normalise incoming data

[UnNormalise][pyearthtools.data.transforms.normalisation.unnormalise] provides the Transforms to unnormalise incoming data

"""

from pyearthtools.data.transforms.normalisation import _utils
from pyearthtools.data.transforms.normalisation.normalise import Normalise
from pyearthtools.data.transforms.normalisation.unnormalise import Unnormalise
