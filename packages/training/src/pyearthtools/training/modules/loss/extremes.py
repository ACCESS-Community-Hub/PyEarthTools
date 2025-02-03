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


import torch
from torch import nn


class ExtremeLoss(nn.Module):
    def __init__(
        self,
        min_weight: float = 0.2,
        max_weight: float = 0.2,
        square_weight: float = 0.6,
        std_weight: float = 0,
    ):
        """
        Extreme Value Loss Function

        Parameters
        ----------
        min_weight, optional
            Weighting applied to difference in min values, by default 0.2
        max_weight, optional
            Weighting applied to difference in max values, by default 0.2
        square_weight, optional
            Weighting applied to difference in MSE, by default 0.6
        std_weight, optional
            Weighting applied to difference in std, by default 0
        """
        super().__init__()

        self.min_weight = min_weight
        self.max_weight = max_weight
        self.square_weight = square_weight
        self.std_weight = std_weight

    def forward(self, output, target):
        mean_target = torch.mean(target)
        square_difference = torch.square(output - target)
        mean_square_difference = torch.mean(square_difference)
        root_mean_square_difference = torch.sqrt(mean_square_difference)

        relative_mean_square_difference = torch.div(root_mean_square_difference, mean_target)

        min_output = torch.min(output)
        min_target = torch.min(target)

        min_difference = torch.abs(min_output - min_target)
        relative_min_difference = torch.div(min_difference, mean_target)

        max_output = torch.max(output)
        max_target = torch.max(target)

        max_difference = torch.abs(max_output - max_target)
        relative_max_difference = torch.div(max_difference, mean_target)

        sd_output = torch.std(output)
        sd_target = torch.std(target)

        sd_difference = torch.abs(sd_output - sd_target)
        relative_sd_difference = torch.div(sd_difference, sd_target)

        loss_value = (
            self.min_weight * relative_min_difference
            + self.max_weight * relative_max_difference
            + self.square_weight * relative_mean_square_difference
            + self.std_weight * relative_sd_difference
        )

        return loss_value
