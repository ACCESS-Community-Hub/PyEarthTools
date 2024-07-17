# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
EDIT Training 

Using `edit` prepare data for training, 
and allow rapid distributed training of Machine Learning Models.
"""
# ruff: noqa: F401

from edit.training import data, wrapper, manage

from edit.training.wrapper import *  # type: ignore # noqa: F403
from edit.training.dataindex import MLDataIndex

# try:
#     from edit.training import modules
# except ImportError:
#     pass

# if __name__ == "__main__":
#     trainer.commands.entry_point()

__version__ = "1.0.1"
