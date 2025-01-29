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
