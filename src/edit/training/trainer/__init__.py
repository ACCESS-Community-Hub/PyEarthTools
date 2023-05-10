"""
!!! Warning
    This subsection is very much still in development, no api structure nor code here is guaranteed.

    This will eventually be full of objects to allow easy training of common deep learning architectures.

## EDIT Training Trainers

Provide simple classes to combine the Data Pipeline defined in [edit.training.data][edit.training.data] and an ML model for training.
"""

from edit.training.trainer.pytorch.trainer import EDITLightningTrainer
from edit.training.trainer import commands
from edit.training.trainer.yaml import from_yaml
from edit.training.trainer.dataindex import MLDataIndex

if __name__ == "__main__":
    commands.entry_point()
