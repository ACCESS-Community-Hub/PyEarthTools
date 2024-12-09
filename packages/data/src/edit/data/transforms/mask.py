# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations
import operator

from typing import Any, Literal, overload

from pathlib import Path

import numpy as np
import xarray as xr

from pyearthtools.data.transforms.transform import Transform
from pyearthtools.data.transforms.utils import parse_dataset

from pyearthtools.utils.decorators import BackwardsCompatibility

OPERATIONS = ["==", "!=", ">", "<", ">=", "<="]
OPERATIONS_TYPE = Literal["==", "!=", ">", "<", ">=", "<="]


def check_operations(operation: OPERATIONS_TYPE | dict[str, OPERATIONS_TYPE]):
    if isinstance(operation, dict):
        for _, val in operation.items():
            check_operations(val)
    else:
        if operation not in OPERATIONS:
            raise KeyError(f"Invalid operation {operation!r}. Must be one of {OPERATIONS}")


class UnderlyingMaskTransform(Transform):
    @overload
    def __filter(
        self,
        data: xr.Dataset,
        value: float | xr.Dataset | np.ndarray,
        replacement_value: xr.Dataset | np.ndarray | float,
        operation: OPERATIONS_TYPE | dict[str, OPERATIONS_TYPE] = "==",
        search_data: xr.Dataset | xr.DataArray | np.ndarray | None = None,
    ) -> xr.Dataset: ...

    @overload
    def __filter(
        self,
        data: xr.DataArray,
        value: float | xr.Dataset | np.ndarray,
        replacement_value: xr.Dataset | np.ndarray | float,
        operation: OPERATIONS_TYPE | dict[str, OPERATIONS_TYPE] = "==",
        search_data: xr.Dataset | xr.DataArray | np.ndarray | None = None,
    ) -> xr.DataArray: ...
    @overload
    def __filter(
        self,
        data: np.ndarray,
        value: float | xr.Dataset | np.ndarray,
        replacement_value: xr.Dataset | np.ndarray | float,
        operation: OPERATIONS_TYPE | dict[str, OPERATIONS_TYPE] = "==",
        search_data: xr.Dataset | xr.DataArray | np.ndarray | None = None,
    ) -> np.ndarray: ...

    def __filter(
        self,
        data: xr.Dataset | xr.DataArray | np.ndarray,
        value: float | xr.Dataset | np.ndarray,
        replacement_value: xr.Dataset | np.ndarray | float,
        operation: OPERATIONS_TYPE | dict[str, OPERATIONS_TYPE] = "==",
        search_data: xr.Dataset | xr.DataArray | np.ndarray | None = None,
    ):
        """
        Mask out data

        Args:
            data (xr.Dataset | np.ndarray):
                Data to apply masked replacement on
            value (float | xr.Dataset | np.ndarray):
                Value to use in conditional statement
            replacement_value (xr.Dataset | np.ndarray | float):
                Value to replace with
            operation (str, optional):
                Operation to mask by. Defaults to "==".
            search_data (xr.Dataset | np.ndarray | None, optional):
                Alternate data find mask on. Defaults to None.
        """
        search_data = search_data or data

        if isinstance(value, str) and value == "nan":
            value = np.nan
        if isinstance(replacement_value, str) and replacement_value == "nan":
            replacement_value = np.nan

        try:
            isnan = np.isnan(value)
        except Exception:
            isnan = False

        operator_package = np
        if isinstance(data, (xr.Dataset, xr.DataArray)):
            operator_package = xr
            data = type(data)(data)  # type: ignore
            if search_data is not None:
                search_data = type(search_data)(search_data)  # type: ignore

        operations_dict = {
            ">": operator.gt,
            ">=": operator.ge,
            "<": operator.lt,
            "<=": operator.le,
        }

        if operation == "==":
            boolean_result = np.isnan(search_data) if isnan else search_data == value
        elif operation == "!=":
            boolean_result = (not np.isnan(search_data)) if isnan else search_data != value
        elif operation in operations_dict:
            boolean_result = operations_dict[operation](search_data, value)
        else:
            raise KeyError(f"Invalid operation: {operation!r}")

        return operator_package.where(boolean_result, replacement_value, data)  # type: ignore

    def filter(
        self,
        data: xr.Dataset | xr.DataArray,
        value: dict | float | str | Path,
        *,
        replacement_value: xr.Dataset | np.ndarray | float | Path | str | dict[str, Any] = np.nan,
        operation: OPERATIONS_TYPE | dict[str, OPERATIONS_TYPE] = "==",
        **kwargs,
    ):
        """
        Run filtering,
        But if any of the given kwargs are dictionaries retrieve the correct element

        Will raise an error if a key is missing from a dictionary when it was present in another
        """

        def get_safely_from_dict(search_key, **kwargs):
            """
            Collapse all dictionarys in kwargs by selecting search_key from them
            """
            return_kwargs = {}
            for key, val in kwargs.items():
                if isinstance(val, dict):
                    val = val[search_key]
                return_kwargs[key] = val
            return return_kwargs

        def get_all_keys(*args):
            """
            Get all keys from all args which are dicts
            """
            keys = []
            for arg in args:
                if isinstance(arg, dict):
                    for key in arg.keys():
                        if key not in keys:
                            keys.append(key)
            return keys

        def parse_str(obj: Any, ds: xr.Dataset | xr.DataArray) -> Any:
            if not isinstance(obj, str):
                return obj

            try:
                return parse_dataset(Path(obj))
            except FileNotFoundError:
                pass

            if isinstance(ds, xr.DataArray):
                ds = ds.to_dataset(name="data")

            from pyearthtools.data.transforms.derive import evaluate

            return evaluate(obj, dataset=ds)

        kwargs = dict(map(lambda x: (x[0], parse_dataset(x[1])), kwargs.items()))

        if any(map(lambda x: isinstance(x, dict), (value, replacement_value, operation))):
            if not isinstance(data, xr.Dataset):
                raise TypeError(
                    "One or more of: 'value', 'replacement_value' or 'operation' was a dictionary, but data was not an xr.Dataset, cannot parse."
                )

            ## If value to look for is dict, get appropriate from dataset keys
            for masking_key in set(get_all_keys(value, replacement_value, operation)).intersection(
                set(list(data.data_vars))
            ):
                try:
                    dict_kwargs = get_safely_from_dict(
                        masking_key,
                        value=value,
                        replacement_value=replacement_value,
                        operation=operation,
                    )
                except KeyError as _:
                    raise KeyError(
                        "A KeyError occured solving the dictionary arguments. Likely a key is missing in one of the dictionary params which was given in another."
                    )
                dict_kwargs = dict(
                    map(
                        lambda x: (x[0], parse_str(x[1], data[masking_key])),
                        dict_kwargs.items(),
                    )
                )
                data[masking_key] = self.__filter(data[masking_key], **dict_kwargs, **kwargs)
            return data

        return self.__filter(
            data,
            parse_dataset(value),
            parse_str(replacement_value, data),
            operation=operation,
            **kwargs,
        )  # type: ignore


class Dataset(UnderlyingMaskTransform):
    def __init__(
        self,
        value: Any,
        reference_dataset: xr.Dataset | str,
        operation: OPERATIONS_TYPE | dict[str, OPERATIONS_TYPE] = "==",
        replacement_value: float | str | xr.Dataset = np.nan,
        squeeze: str | list = "None",
    ):
        """
        Mask data using a reference dataset

        Will replace data on incoming dataset where condition is met on `reference_dataset`

        Args:
            reference_dataset (xr.Dataset | str | dict):
                Reference dataset to calculate mask from.
                Can be dataset, str as Path, or a dictionary referencing incoming data variables
                containing the prior types.
            value (Any, optional):
                Value to mask at.
                Can be array, dataset, string or dictionary.
                Defaults to np.NaN.
            operation (Literal['==', '!=', '>', '<', '>=','<='] | dict, optional):
                Criteria to search by. Can be dictionary for dataset keys. Defaults to "==".
            replacement_value (float | str | xr.Dataset | dict, optional):
                Value to replace with. Can be str pointing to dataset or dataset itself, or a dictionary.
                Defaults to np.nan
            squeeze (str | list, optional):
                Dims to squeeze on reference dataset. Defaults to 'None'

        Returns:
            (Transform): Transform to apply mask to data
        """
        super().__init__()
        self.record_initialisation()

        check_operations(operation)
        self._reference_dataset = reference_dataset
        self._value = value
        self._operation = operation
        self._replacement_value = replacement_value
        self._squeeze = squeeze

    def apply(self, dataset: xr.Dataset) -> xr.Dataset:
        if not isinstance(dataset, (xr.Dataset, xr.DataArray)):
            raise TypeError(f"Must be an xarray object, not {type(dataset)}")

        if isinstance(self._reference_dataset, dict):
            valid_keys = ((key, val) for key, val in self._reference_dataset.items() if key in dataset)
            ## Parse and filter all keys with reference dataset
            for key, ref_val in valid_keys:
                ref_val = parse_dataset(ref_val)
                dataset[key] = self.filter(
                    dataset[key],
                    value=self._value,
                    replacement_value=self._replacement_value,
                    operation=self._operation,  # type: ignore
                    search_data=ref_val,
                )
                return dataset

        parsed_reference_dataset = parse_dataset(self._reference_dataset)

        if not isinstance(parsed_reference_dataset, xr.Dataset):
            raise TypeError(f"'reference_dataset' must be xr.Dataset, not {type(parsed_reference_dataset)}")

        parsed_reference_dataset = parsed_reference_dataset.squeeze(self._squeeze).compute()
        return self.filter(
            dataset,
            value=self._value,
            replacement_value=self._replacement_value,
            operation=self._operation,  # type: ignore
            search_data=parsed_reference_dataset,
        )


@BackwardsCompatibility(Dataset)
def dataset(*a, **k): ...


class Replace(UnderlyingMaskTransform):
    def __init__(
        self,
        value: dict | float | str,
        operation: OPERATIONS_TYPE | dict[str, OPERATIONS_TYPE] = "==",
        replacement_value: float | dict[str, Any] | str | xr.Dataset = np.nan,
    ):
        """
        Replace Values in dataset with replacement_value when matching criteria

        Args:
            value (dict | float | str):
                Value to mask at.
                Can be array, dataset, string or dictionary.
                Dictionary refers to variables and values.
            operation (Literal['==', '!=', '>', '<', '>=','<='] | dict, optional):
                Criteria to search by. Can be dictionary for dataset keys. Defaults to "==".
            replacement_value (float | str | xr.Dataset | dict, optional):
                Value to replace with. Can be str pointing to dataset or dataset itself, or a dictionary.
                Defaults to np.nan

        Raises:
            KeyError: If invalid operation is provided

        """
        super().__init__()
        self.record_initialisation()

        check_operations(operation)
        self._value = value
        self._operation = operation
        self._replacement_value = replacement_value

    def apply(self, data):
        return self.filter(
            data,
            value=self._value,
            replacement_value=self._replacement_value,
            operation=self._operation,  # type: ignore
        )


@BackwardsCompatibility(Replace)
def replace_value(*a, **k): ...


__all__ = ["Dataset", "Replace"]
