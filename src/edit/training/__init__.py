"""
EDIT Training 

Using [edit.data][edit.data.indexes] DataIndexes prepare data for training, 
and allow rapid distributed training of Machine Learning Models.

"""

from edit.training import loader, trainer
from edit.training.trainer import MLDataIndex

try:
    from edit.training import modules
except ImportError:
    pass


from_dict = from_yaml
load = from_yaml

if __name__ == "__main__":
    trainer.commands.entry_point()

__version__ = '2023.12.1'