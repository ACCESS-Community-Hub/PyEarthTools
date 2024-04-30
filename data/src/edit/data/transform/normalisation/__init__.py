# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

"""
A sophisticated [Transform][edit.data.transform.Transform] to normalise and unnormalise data.

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
[Normalise][edit.data.transform.normalisation.normalise] provides the Transforms to normalise incoming data

[UnNormalise][edit.data.transform.normalisation.unnormalise] provides the Transforms to unnormalise incoming data

"""

from edit.data.transform.normalisation import _utils
from edit.data.transform.normalisation.normalise import Normalise
from edit.data.transform.normalisation.unnormalise import Unnormalise
