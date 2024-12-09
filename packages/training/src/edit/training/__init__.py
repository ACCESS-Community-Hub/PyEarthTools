# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
pyearthtools Training 

Using `pyearthtools` prepare data for training, 
and allow rapid distributed training of Machine Learning Models.
"""
# ruff: noqa: F401

from pyearthtools.training import logger as _

from pyearthtools.training import data, wrapper, manage

from pyearthtools.training.wrapper import *  # type: ignore # noqa: F403
from pyearthtools.training.dataindex import MLDataIndex

__version__ = "0.1.0"

# try:
#     from pyearthtools.training import modules
# except ImportError:
#     pass

