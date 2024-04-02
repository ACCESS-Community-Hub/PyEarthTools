# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

import functools
from typing import Callable, Union
from torch import nn
import numpy as np
import torch
import math


def sin(m_adjust=1, drop_rate=0.5, dimensions: tuple[int] = [-2, -1]):
    def _sin_func(*args):
        x = args[dimensions[0]]
        y = args[dimensions[1]]

        size = (np.max(x), np.max(y))

        x_value = np.sin(x * ((2 * np.pi) / (2 * size[0]))) * m_adjust
        y_value = np.sin(y * ((2 * np.pi) / (2 * size[1]))) * m_adjust

        return ((x_value + y_value) / 2) ** drop_rate

    return _sin_func


def gaussian(std_dev, dimensions: tuple[int] = [-2, -1]):
    set_at_1 = std_dev * 2.5063

    def _guassian(*args):
        x = args[dimensions[0]]
        y = args[dimensions[1]]

        size = (np.max(x), np.max(y))

        dist = (
            lambda x, u: 1
            / (std_dev * math.sqrt(2 * math.pi))
            * np.power(np.e, -1 / 2 * ((x - u) / std_dev) ** 2)
            * set_at_1
        )

        x_value = dist(x, size[0] / 2)
        y_value = dist(y, size[1] / 2)

        return np.minimum(x_value, y_value)

    return _guassian


class centre_weighted(nn.Module):
    def __init__(
        self,
        torch_loss_function: Union[Callable, str],
        centre_function: str = "sin",
        function_kwargs: dict = {},
        *,
        min_value: float = 0,
        max_value: float = 1,
        dimensions: tuple[int] = [-2, -1],
        **kwargs,
    ):
        """
        Centre weighted loss function.

        Apply centre biased weights to output from a torch loss function.

        Parameters
        ----------
        torch_loss_function
            Normal torch function to find loss
        centre_function, optional
            Function to create centering, by default sin
            Either sin, or gaussian
        *function_kwargs, optional
            Changes to function kwargs
        min_value, optional
            Min clip value, by default 0
        max_value, optional
            Max clip value, by default 1
        dimensions, optional
            Which dimensions to use as 'spatial' to make centre, by default [-2,-1]
        """
        super().__init__()
        if isinstance(torch_loss_function, str):
            torch_loss_function = getattr(nn, torch_loss_function)
            torch_loss_function = torch_loss_function(**kwargs)

        self.loss = torch_loss_function

        self.min_value = min_value
        self.max_value = max_value

        self.dimensions = dimensions

        if centre_function == "sin":
            self.centre_func = sin(**function_kwargs)
        elif centre_function == "gaussian":
            self.centre_func = gaussian(**function_kwargs)

    @functools.lru_cache(None)
    def make_weight_array(self, shape: tuple):
        """
        Create Weight Array with shape

        Parameters
        ----------
        shape
            Shape to create weight array with

        Returns
        -------
            np.array of weights
        """
        return np.fromfunction(self.centre_func, shape).clip(self.min_value, self.max_value)

    def forward(self, output, target):
        weight = torch.Tensor(self.make_weight_array(target.shape)).to(target)
        loss = self.loss(output, target)
        loss = loss * weight
        return loss.sum() / weight.sum()
