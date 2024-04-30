# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

from __future__ import annotations
import operator

from typing import Any, Literal, Union

from pathlib import Path

import numpy as np
import xarray as xr

import edit.data
from edit.data.transform.transform import Transform
from edit.data.transform.utils import parse_dataset

OPERATIONS = ["==", "!=", ">", "<", ">=", "<="]


def check_operations(operation: str | dict):
    if isinstance(operation, dict):
        for _, val in operation.items():
            check_operations(val)
    else:
        if operation not in OPERATIONS:
            raise KeyError(f"Invalid operation {operation!r}. Must be one of {OPERATIONS}")


class UnderlyingMaskTransform(Transform):
    def __filter(
        self,
        data: xr.Dataset,
        value: float | xr.Dataset | np.ndarray,
        replacement_value: xr.Dataset | np.ndarray | float,
        operation: str = "==",
        search_data: xr.Dataset | np.ndarray | None = None,
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
            data = type(data)(data)
            search_data = type(search_data)(search_data)

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

        return operator_package.where(boolean_result, replacement_value, data)

        

    def filter(
        self,
        data: xr.Dataset | xr.DataArray,
        value: dict | float | str | Path,
        *,
        replacement_value: xr.Dataset | np.ndarray | float | Path = np.nan,
        operation: str | dict = "==",
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

        def parse_str(obj: Any, ds: xr.Dataset) -> Any:
            if not isinstance(obj, str):
                return obj

            try:
                return parse_dataset(Path(obj))
            except:
                pass
            if isinstance(ds, xr.DataArray):
                ds = ds.to_dataset(name="data")
            from edit.data.transform.derive import evaluate

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
                except KeyError as e:
                    raise KeyError(
                        f"A KeyError occured solving the dictionary arguments. Likely a key is missing in one of the dictionary params which was given in another."
                    )
                dict_kwargs = dict(map(lambda x: (x[0], parse_str(x[1], data[masking_key])), dict_kwargs.items()))
                data[masking_key] = self.__filter(data[masking_key], **dict_kwargs, **kwargs)
            return data

        return self.__filter(
            data,
            parse_dataset(value),
            parse_str(replacement_value, data),
            operation=operation,
            **kwargs,
        )

    @property
    def __doc__(self):
        return f"Mask data given arguments"


class MaskTransform:
    def __new__(cls, value, *args, **kwargs) -> Transform:
        if isinstance(value, str) and value == "nan":
            value = np.nan
        instance_check = value
        if isinstance(instance_check, dict):
            instance_check = instance_check[list(instance_check.keys())[0]]

        if isinstance(instance_check, (str, xr.Dataset)):
            return cls.dataset(value, *args, **kwargs)
        elif isinstance(instance_check, (int, float)) or value == np.nan:
            return cls.replace_value(value, *args, **kwargs)
        raise TypeError(f"Unable to assign method for {value!r}")

    @classmethod
    def dataset(
        cls,
        value: Any,
        reference_dataset: xr.Dataset | str,
        operation: "Literal[OPERATIONS]" = "==",
        replacement_value: float | str | xr.Dataset = np.nan,
        squeeze: str | list = "None",
    ) -> Transform:
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

        check_operations(operation)

        class MaskingTransform(UnderlyingMaskTransform):
            """Mask data based on reference dataset"""

            @property
            def _info_(self):
                return dict(
                    reference_dataset=reference_dataset,
                    operation=operation,
                    value=value,
                    replacement_value=replacement_value,
                    squeeze=squeeze,
                )

            def apply(self, dataset: xr.Dataset) -> xr.Dataset:
                if not isinstance(dataset, (xr.Dataset, xr.DataArray)):
                    raise TypeError(f"Must be an xarray object, not {type(dataset)}")

                if isinstance(reference_dataset, dict):
                    valid_keys = ((key, val) for key, val in reference_dataset.items() if key in dataset)
                    ## Parse and filter all keys with reference dataset
                    for key, ref_val in valid_keys:
                        ref_val = parse_dataset(ref_val)
                        dataset[key] = self.filter(
                            dataset[key],
                            value=value,
                            replacement_value=replacement_value,
                            operation=operation,
                            search_data=ref_val,
                        )
                        return dataset

                parsed_reference_dataset = parse_dataset(reference_dataset)

                if not isinstance(parsed_reference_dataset, xr.Dataset):
                    raise TypeError(f"'reference_dataset' must be xr.Dataset, not {type(parsed_reference_dataset)}")

                parsed_reference_dataset = parsed_reference_dataset.squeeze(squeeze).compute()
                return self.filter(
                    dataset,
                    value=value,
                    replacement_value=replacement_value,
                    operation=operation,
                    search_data=parsed_reference_dataset,
                )

        return MaskingTransform()

    @classmethod
    def replace_value(
        cls,
        value: dict | float | str,
        operation: "Literal[OPERATIONS] | dict" = "==",
        replacement_value: float | dict | str | xr.Dataset = np.nan,
    ) -> Transform:
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

        Returns:
            Transform: Transform to mask dataset
        """
        check_operations(operation)

        class MaskingTransform(UnderlyingMaskTransform):
            """Mask data from values"""

            @property
            def _info_(self):
                return dict(
                    value=value,
                    operation=operation,
                    replacement_value=replacement_value,
                )

            def apply(self, data: xr.Dataset) -> xr.Dataset:
                return self.filter(
                    data,
                    value=value,
                    replacement_value=replacement_value,
                    operation=operation,
                )

        return MaskingTransform()
