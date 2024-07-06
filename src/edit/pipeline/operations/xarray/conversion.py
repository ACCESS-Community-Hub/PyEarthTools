# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

# type: ignore[reportPrivateImportUsage]


from typing import Optional, Union, TypeVar
from pathlib import Path

import numpy as np
import xarray as xr

from edit.utils.data import converter

from edit.pipeline.operation import Operation

XARRAY_OBJECTS = TypeVar("XARRAY_OBJECTS", xr.Dataset, xr.DataArray)
FILE_TYPES = Union[str, Path]

__all__ = ["ToNumpy"]


class ToNumpy(Operation):
    """
    Convert xarray objects to np.ndarray's
    """

    _override_interface = "Serial"

    def __init__(
        self,
        reference_dataset: Optional[FILE_TYPES] = None,
        saved_records: Optional[FILE_TYPES] = None,
        run_parallel: bool = False,
    ):
        """DataOperation to convert data to [np.array][numpy.ndarray]

        If speed is needed without an `undo`, set `run_parallel` to True, and split the data into seperate
        datasets as much as possible.
            `edit.pipeline.operations.xarray.split.OnVariables()` can be useful here

        Args:
            reference_dataset (Optional[FILE_TYPES], optional):
                Reference dataset to run through numpy converter to initialise converter.
                Will be overwritten when this is given a dataset.
                Defaults to None.
            saved_records (Optional[FILE_TYPES], optional):
                Saved records to set numpy converter with.
                Will be overwritten when this is given a dataset.
                Defaults to None.
            run_parallel (bool, optional):
                Whether to run in parallel, will cause `undo` to fail without `saved_records`.
                If an undo pipeline is needed, set this to False.
                Defaults to False.
        """
        super().__init__(
            recognised_types={"apply": (xr.Dataset, xr.DataArray, tuple), "undo": (np.ndarray,)},
            split_tuples=False,
        )
        self.record_initialisation()

        self._numpy_converter = converter.NumpyConverter()
        self._saved_records = saved_records
        self._reference_dataset = reference_dataset
        self._converters = []
        self._run_parallel = run_parallel

        if reference_dataset and saved_records:
            raise ValueError("Cannot provide both `reference_dataset` and `saved_records`.")

        def make_converter() -> converter.NumpyConverter:
            numpy_converter = converter.NumpyConverter()
            if saved_records:
                numpy_converter.load_records(saved_records)
            if reference_dataset:
                numpy_converter.convert_xarray_to_numpy(xr.open_dataset(reference_dataset), replace=True)
            return numpy_converter

        self._make_converter = make_converter

    def _get_converters(self, number: int) -> tuple[converter.NumpyConverter]:
        """
        Retrieve a set number of NumpyConverter, creating new ones if needed
        """
        return_values = []

        for i in range(number):
            if i < len(self._converters):
                return_values.append(self._converters[i])
            else:
                self._converters.append(self._make_converter())
                return_values.append(self._converters[-1])

        return tuple(return_values)

    def apply_func(self, sample: Union[tuple[XARRAY_OBJECTS, ...], XARRAY_OBJECTS]):
        if isinstance(sample, tuple) and self._run_parallel:

            def run_converter(sub_samp: XARRAY_OBJECTS, converter: converter.NumpyConverter):
                return converter.convert_from_xarray(sub_samp)

            parallel_interface = self.get_parallel_interface(["Delayed", "Serial"])
            return tuple(
                parallel_interface.collect(
                    parallel_interface.map(
                        lambda x: run_converter(*x), tuple(zip(sample, self._get_converters(len(sample))))
                    )
                )
            )

        result = self._get_converters(1)[0].convert_from_xarray(sample, replace=True)
        if self._saved_records:
            self._numpy_converter.save_records(self._saved_records)
        return result

    def undo_func(self, sample: Union[tuple[np.ndarray, ...], np.ndarray]):
        return self._get_converters(1)[0].convert_to_xarray(sample, pop=False)


class ToDask(Operation):
    """
    Convert xarray objects to pure dask arrays
    """

    _override_interface = "Serial"

    def __init__(self):
        import dask.array as da

        super().__init__(
            split_tuples=True,
            recursively_split_tuples=True,
            recognised_types=dict(
                apply=(xr.Dataset, xr.DataArray),
                undo=(
                    da.Array,
                    np.ndarray,
                ),
            ),
        )
        self.record_initialisation()
        self._converter = converter.DaskConverter()

    def apply_func(self, sample: XARRAY_OBJECTS):
        return self._converter.convert_from_xarray(sample, replace=True)

    def undo_func(self, sample):
        return self._converter.convert_to_xarray(sample, pop = False)
