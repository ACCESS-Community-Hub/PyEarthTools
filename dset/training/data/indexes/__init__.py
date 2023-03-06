"""
Data Indexes designed for ML use, using dset.data
"""

from dset.training.data.indexes.data_interface import Data_Interface as DataInterface
from dset.training.data.indexes.indexes import (
    AsNumpy,
    CombineDataIndex,
    PatchingDataIndex,
    PatchingUpdate,
)
from dset.training.data.indexes.patching import Tesselator
