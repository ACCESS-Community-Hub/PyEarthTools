# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.


from typing import Optional, Union
from pathlib import Path

import numpy as np
import xarray as xr

from edit.utils.data import NumpyConverter

from edit.pipeline_V2.operation import Operation

XARRAY_OBJECTS = Union[xr.Dataset, xr.DataArray]


class ToNumpy(Operation):
    """
    Operation to convert data to [np.array][numpy.ndarray]
    """

    def __init__(
        self, reference_dataset: Optional[Union[str, Path]] = None, saved_records: Optional[Union[str, Path]] = None
    ):
        """DataOperation to convert data to [np.array][numpy.ndarray]

        Args:
            reference_dataset (Optional[Union[str, Path]], optional):
                Reference dataset to run through numpy converter to initialise converter.
                Will be overwritten when this is given a dataset.
                Defaults to None.
            saved_records (Optional[Union[str, Path]], optional):
                Saved records to set numpy converter with.
                Will be overwritten when this is given a dataset.
                Defaults to None.
        """
        super().__init__(
            recognised_types={"apply": (xr.Dataset, xr.DataArray, tuple), "undo": (np.ndarray,)},
            split_tuples=False,
        )
        self.record_initialisation()

        self._numpy_converter = NumpyConverter()

        if saved_records:
            self._numpy_converter.load_records(saved_records)
        if reference_dataset:
            self._numpy_converter.convert_xarray_to_numpy(xr.open_dataset(reference_dataset), replace=True)

    def apply_func(self, sample: Union[tuple[XARRAY_OBJECTS, ...], XARRAY_OBJECTS]):
        return self._numpy_converter.convert_xarray_to_numpy(sample, replace=True)

    def undo_func(self, sample: Union[tuple[np.ndarray, ...], np.ndarray]):
        return self._numpy_converter.convert_numpy_to_xarray(sample, pop=False)
