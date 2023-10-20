"""
Collection of Dataloaders to use at the end of the pipeline to get data for ML

| Name                | Description |
| ------------------- | ----------- |
| [PytorchIterable][edit.training.loader.pytorch]     | Basic PyTorch IterableDataset |
| [CustomLoader][edit.training.loader.custom]     | Basic Custom DataLoader to batch data |
"""

try:
    from edit.training.loader.pytorch import PytorchIterable
except ImportError:
    pass
from edit.training.loader.custom import CustomLoader

# from .dali import DALILoader
