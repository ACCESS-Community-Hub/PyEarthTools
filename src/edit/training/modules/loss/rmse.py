import torch

class RMSELoss(torch.nn.MSELoss):
    def __init__(self) -> None:
        self.MSELoss = torch.nn.MSELoss()

    def forward(self, output, target):
        mseloss = self.MSELoss(output, target)
        return torch.sqrt(mseloss)