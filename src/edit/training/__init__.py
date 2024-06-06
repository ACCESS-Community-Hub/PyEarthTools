# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
EDIT Training 

Using [edit.data][edit.data.indexes] DataIndexes prepare data for training, 
and allow rapid distributed training of Machine Learning Models.

"""

from edit.training import loader, trainer, manage
from edit.training.trainer import MLDataIndex, from_yaml

try:
    from edit.training import modules
except ImportError:
    pass


from_dict = from_yaml
load = from_yaml

if __name__ == "__main__":
    trainer.commands.entry_point()

__version__ = "2024.05.01"
