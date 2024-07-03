# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
Remapping Operations.
"""

from abc import abstractmethod, ABCMeta
from typing import Type, TypeVar
import xarray as xr

from edit.pipeline import Operation

XR_TYPE = TypeVar('XR_TYPE', xr.Dataset, xr.DataArray)

class BaseRemap(Operation, metaclass = ABCMeta):
    """
    Base class for remappers.
    """

    def __init__(self, *, split_tuples: bool = True, recursively_split_tuples: bool = True, recognised_types: tuple[Type, ...] = (xr.Dataset, xr.DataArray)):
        super().__init__(split_tuples=split_tuples, recursively_split_tuples=recursively_split_tuples, recognised_types=recognised_types)
        
    def apply_func(self, sample):
        return self.remap(sample)
    
    def undo_func(self, sample):
        return self.inverse_remap(sample)

    @abstractmethod
    def remap(self, sample: XR_TYPE) -> XR_TYPE:
        """
        Forward mapping operation. Must be defined in subclasses.
        """
        raise NotImplementedError()

    @abstractmethod
    def inverse_remap(self, sample: XR_TYPE) -> XR_TYPE:
        """
        Inverse mapping operation. Must be defined in subclasses.
        """
        raise NotImplementedError()
