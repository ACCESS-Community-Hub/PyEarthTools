"""
Collection of Dataloaders to use at the end of the Pipeline

| Name                | Description |
| ------------------- | ----------- |
| [PytorchIterable][edit.pipeline.loaders.pytorch]     | Basic PyTorch IterableDataset |
| [CustomLoader][edit.pipeline.loaders.custom]     | Basic Custom DataLoader to batch data |
"""

from edit.pipeline.loaders.pytorch import PytorchIterable
from edit.pipeline.loaders.custom import CustomLoader

# from .dali import DALILoader
