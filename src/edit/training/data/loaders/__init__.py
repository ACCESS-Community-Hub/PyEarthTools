"""
Collection of Dataloaders to use at the end of the Pipeline

| Name                | Description |
| ------------------- | ----------- |
| [PytorchIterable][edit.training.data.loaders.pytorch]     | Basic PyTorch IterableDataset |
| [CustomLoader][edit.training.data.loaders.custom]     | Basic Custom DataLoader to batch data |
"""

from edit.training.data.loaders.pytorch import PytorchIterable
from edit.training.data.loaders.custom import CustomLoader
from .climax import ClimaXDataLoader

# from .dali import DALILoader
