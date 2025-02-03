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


from torch import nn
import torch
import pyearthtools.training


class ComponentLoss(nn.Module):
    """
    Loss function made of multiple components


    Example
        >>> loss = ComponentLoss([0.5, 0.5], MSELoss = {}, SSIMLoss = {})
        # Even weighting of MSE and SSIM
    """

    def __init__(self, weights: list[float], **loss: dict[str, dict]) -> None:
        """
        Setup loss function

        Args:
            weights (list[float]):
                Weights for each loss function.
            loss (dict[str, dict], optional):
                Loss name and init args
                For each loss, the key will be used to find and load the loss, and the value used to initalise.
                If no initalisation is required, set to {}
        """
        super().__init__()

        if not loss:
            raise ValueError("Component losses cannot be empty")

        if weights is None:
            raise TypeError(f"Weights must be provided")
            weights = []
            for key, value in loss.items():
                if not isinstance(value, dict):
                    raise TypeError(f"Each loss must contain a dict for init args, cannot be {type(value)} ")

                if isinstance(value, dict) and "weight" not in value:
                    raise KeyError(
                        f"If `weights` are not specified, each loss must contain a `weight` key. {key} does not. {value.keys()}"
                    )
                weights.append(value.pop("weight"))

        if not len(weights) == len(list(loss.keys())):
            raise TypeError(
                f"`weights` must contain the same number of elements as `loss`. {len(weights)} != {len(list(loss.keys()))}"
            )

        self.weights = nn.Parameter(torch.Tensor(weights), False)
        self.losses = nn.ModuleList(pyearthtools.training.modules.get_loss(key, **value) for key, value in loss.items())

    def forward(self, output, target):
        loss = None
        for i, loss_module in enumerate(self.losses):
            element_loss = loss_module(output, target)

            if loss is None:
                loss = element_loss * self.weights[i]
            else:
                loss += element_loss * self.weights[i]

        return loss
