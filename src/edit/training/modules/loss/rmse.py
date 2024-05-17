# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

import torch


class RMSELoss(torch.nn.MSELoss):
    def __init__(self) -> None:
        super().__init__()
        self.MSELoss = torch.nn.MSELoss()

    def forward(self, output, target):
        mseloss = self.MSELoss(output, target)
        return torch.sqrt(mseloss)
