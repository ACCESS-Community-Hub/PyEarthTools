# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

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
