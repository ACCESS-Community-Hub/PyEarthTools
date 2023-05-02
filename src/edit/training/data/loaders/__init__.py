"""
Collection of Dataloaders to use at the end of the Pipeline

| Name                | Description |
| ------------------- | ----------- |
| [PytorchIterable][edit.training.data.loaders.pytorch]     | Basic PyTorch IterableDataset |
"""

from edit.training.data.loaders.pytorch import PytorchIterable
from .climax import ClimaXDataLoader

# from .dali import DALILoader
