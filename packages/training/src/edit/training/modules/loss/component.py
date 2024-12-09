# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

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
