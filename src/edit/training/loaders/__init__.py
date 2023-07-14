"""
Collection of Dataloaders to use at the end of the training

| Name                | Description |
| ------------------- | ----------- |
| [PytorchIterable][edit.training.loaders.pytorch]     | Basic PyTorch IterableDataset |
| [CustomLoader][edit.training.loaders.custom]     | Basic Custom DataLoader to batch data |
"""

from edit.training.loaders.pytorch import PytorchIterable
from edit.training.loaders.custom import CustomLoader

# from .dali import DALILoader
