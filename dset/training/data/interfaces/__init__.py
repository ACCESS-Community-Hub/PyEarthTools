"""
Data Indexes designed for ML use, using dset.data
"""

from dset.training.data.interfaces.data_interface import Data_Interface as DataInterface
from dset.training.data.interfaces.indexes import (
    AsNumpy,
    CombineDataIndex,
    PatchingDataIndex,
)
from dset.training.data.interfaces.patching import Tesselator
